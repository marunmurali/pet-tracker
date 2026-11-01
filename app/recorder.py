"""Ring buffers of recent video (as JPEG) and audio, and a clip writer (H.264 + AAC via ffmpeg).

A clip = everything in the ring buffer from `pre_s` before the trigger + `post_s` after it.
Frames are kept JPEG-compressed so a few seconds of history costs a few MB, not hundreds.
"""
from __future__ import annotations

import collections
import logging
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

log = logging.getLogger(__name__)

AUDIO_RATE = 16000


@dataclass
class _Capture:
    start: float
    end: float
    path: Path
    on_done: Callable[[Path | None], None]
    frames: list = field(default_factory=list)
    audio: list = field(default_factory=list)


class Recorder:
    def __init__(self, clips_dir: Path, pre_s: float, post_s: float, jpeg_quality: int = 80):
        self.clips_dir = clips_dir
        self.pre_s, self.post_s = pre_s, post_s
        self.jpeg_quality = jpeg_quality
        self._frames: collections.deque = collections.deque()  # (ts, jpeg bytes)
        self._audio: collections.deque = collections.deque()  # (ts, int16 array)
        self._captures: list[_Capture] = []
        self._writers: list[threading.Thread] = []
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
            self._audio.clear()

    @property
    def busy(self) -> bool:
        return bool(self._captures)

    def add_frame(self, ts: float, frame: np.ndarray) -> None:
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return
        item = (ts, jpg.tobytes())
        with self._lock:
            self._frames.append(item)
            _trim(self._frames, ts - self.pre_s - 1)
            for c in self._captures:
                c.frames.append(item)
        self._finish_due(ts)

    def add_audio(self, ts: float, samples: np.ndarray) -> None:
        item = (ts, samples.copy())
        with self._lock:
            self._audio.append(item)
            _trim(self._audio, ts - self.pre_s - 1)
            for c in self._captures:
                c.audio.append(item)

    def trigger(self, ts: float, name: str, on_done: Callable[[Path | None], None],
                pre_s: float | None = None, post_s: float | None = None) -> Path:
        """Start a clip around `ts`. `on_done(path or None)` is called from a worker thread."""
        pre = self.pre_s if pre_s is None else pre_s
        post = self.post_s if post_s is None else post_s
        path = self.clips_dir / f"{name}.mp4"
        with self._lock:
            c = _Capture(ts - pre, ts + post, path, on_done)
            c.frames = [f for f in self._frames if f[0] >= c.start]
            c.audio = [a for a in self._audio if a[0] >= c.start]
            self._captures.append(c)
        return path

    def abort(self) -> None:
        """Finish in-progress clips with what they have (e.g. the stream is being switched off)."""
        self._finish_due(float("inf"))

    def _finish_due(self, now: float) -> None:
        with self._lock:
            due = [c for c in self._captures if now >= c.end]
            self._captures = [c for c in self._captures if now < c.end]
        for c in due:
            t = threading.Thread(target=self._write, args=(c,), name="clip-writer", daemon=True)
            t.start()
            self._writers = [w for w in self._writers if w.is_alive()] + [t]

    def wait(self, timeout: float = 60) -> None:
        """Wait for clips that are being encoded (used at shutdown)."""
        deadline = time.monotonic() + timeout
        for t in list(self._writers):
            t.join(max(0.0, deadline - time.monotonic()))

    def _write(self, c: _Capture) -> None:
        path = None
        try:
            if write_clip(c.path, c.frames, c.audio):
                path = c.path
        except Exception:
            log.exception("clip write failed")
        try:
            c.on_done(path)
        except Exception:
            log.exception("clip callback failed")


def _trim(dq: collections.deque, oldest: float) -> None:
    while dq and dq[0][0] < oldest:
        dq.popleft()


def write_clip(path: Path, frames: list, audio: list) -> bool:
    """frames: [(ts, jpeg)], audio: [(ts, int16 array)] -> MP4 (H.264 + AAC). Returns True on success."""
    if len(frames) < 2:
        log.warning("not enough frames for a clip (%d)", len(frames))
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = frames[-1][0] - frames[0][0]
    fps = max(1.0, (len(frames) - 1) / duration) if duration > 0 else 10.0
    tmp = path.with_suffix(".tmp.mp4")
    with tempfile.TemporaryDirectory() as td:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", f"{fps:.3f}", "-i", "pipe:0"]
        if audio:
            wav = Path(td) / "a.wav"
            # audio chunks are stamped at arrival (end of chunk); align the track start to the first frame
            first_audio_start = audio[0][0] - len(audio[0][1]) / AUDIO_RATE
            offset = first_audio_start - frames[0][0]
            pcm = np.concatenate([a[1] for a in audio])
            if offset > 0:
                pcm = np.concatenate([np.zeros(int(offset * AUDIO_RATE), np.int16), pcm])
            elif offset < 0:
                pcm = pcm[int(-offset * AUDIO_RATE):]
            with wave.open(str(wav), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(AUDIO_RATE)
                w.writeframes(pcm.astype(np.int16).tobytes())
            cmd += ["-i", str(wav), "-c:a", "aac", "-b:a", "64k", "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(tmp)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for _, jpg in frames:
                proc.stdin.write(jpg)
            proc.stdin.close()
        except BrokenPipeError:
            pass
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait(timeout=120) != 0:
            log.error("ffmpeg clip encode failed: %s", err.strip()[-500:])
            tmp.unlink(missing_ok=True)
            return False
    tmp.rename(path)
    log.info("clip saved: %s (%.1f s, %d frames)", path.name, duration, len(frames))
    return True


def clip_name(kind: str, ts: float | None = None) -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(ts or time.time()))}-{kind}"

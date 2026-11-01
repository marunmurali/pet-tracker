"""Single ffmpeg process per connection: raw BGR video on stdout, 16 kHz mono s16le audio on an extra pipe.

The Tapo only allows a couple of concurrent RTSP sessions, so video and audio share one connection.
`stop()` kills ffmpeg, which closes the RTSP session entirely ("off means off").
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

AUDIO_RATE = 16000
AUDIO_CHUNK_S = 0.1  # audio is delivered in 100 ms chunks

FrameCallback = Callable[[float, np.ndarray], None]
AudioCallback = Callable[[float, np.ndarray], None]


def build_ffmpeg_cmd(source: str, width: int, height: int, fps: int, audio_fd: int, realtime: bool) -> list[str]:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if source.startswith("rtsp://"):
        cmd += ["-rtsp_transport", "tcp", "-timeout", "10000000", "-fflags", "nobuffer", "-flags", "low_delay"]
    elif realtime:
        cmd += ["-re", "-stream_loop", "-1"]  # local file: play at real speed, loop forever
    cmd += [
        "-i", source,
        "-map", "0:v:0", "-vf", f"fps={fps},scale={width}:{height}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        # aresample=async smooths the Tapo's jittery G.711 timestamps (otherwise ffmpeg logs dts errors)
        "-map", "0:a:0?", "-af", "aresample=async=1:first_pts=0", "-ac", "1", "-ar", str(AUDIO_RATE),
        "-f", "s16le", f"pipe:{audio_fd}",
    ]
    return cmd


def redact(text: str) -> str:
    """Hide credentials in rtsp://user:pass@host URLs before logging."""
    return re.sub(r"(rtsp://)[^@/\s]+@", r"\1***@", text)


class Stream:
    def __init__(self, source: str, width: int, height: int, fps: int,
                 on_frame: FrameCallback, on_audio: AudioCallback, realtime: bool = True):
        self.source = source
        self.width, self.height, self.fps = width, height, fps
        self.on_frame, self.on_audio = on_frame, on_audio
        self.realtime = realtime
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.connected = False
        self.last_frame_ts = 0.0

    # --- lifecycle -------------------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="stream", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            self._kill()
            t, self._thread = self._thread, None
        if t:
            t.join(timeout=5)
        self.connected = False

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # --- internals ---------------------------------------------------------------------------------
    def _kill(self) -> None:
        p = self._proc
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            started = time.monotonic()
            self._run_once()
            self.connected = False
            if self._stop.is_set():
                break
            if time.monotonic() - started > 30:
                backoff = 1.0  # it was a healthy connection that dropped; retry quickly
            log.warning("stream ended; reconnecting in %.0f s", backoff)
            self._stop.wait(backoff)
            backoff = min(backoff * 2, 60)

    def _run_once(self) -> None:
        r_fd, w_fd = os.pipe()
        cmd = build_ffmpeg_cmd(self.source, self.width, self.height, self.fps, w_fd, self.realtime)
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          pass_fds=(w_fd,), bufsize=0)
        except OSError as e:
            log.error("cannot start ffmpeg: %s", e)
            os.close(r_fd)
            os.close(w_fd)
            return
        os.close(w_fd)  # only ffmpeg keeps the write end; we get EOF when it exits
        proc = self._proc
        audio_file = os.fdopen(r_fd, "rb", buffering=0)
        audio_t = threading.Thread(target=self._read_audio, args=(audio_file,), name="stream-audio", daemon=True)
        err_t = threading.Thread(target=self._read_stderr, args=(proc,), name="stream-stderr", daemon=True)
        audio_t.start()
        err_t.start()
        try:
            self._read_video(proc)
        finally:
            self._kill()
            audio_t.join(timeout=3)
            audio_file.close()
            err_t.join(timeout=3)

    def _read_video(self, proc: subprocess.Popen) -> None:
        frame_bytes = self.width * self.height * 3
        stdout = proc.stdout
        assert stdout is not None
        while not self._stop.is_set():
            buf = _read_exact(stdout, frame_bytes)
            if buf is None:
                return
            if not self.connected:
                log.info("stream connected (%dx%d @ %d fps)", self.width, self.height, self.fps)
                self.connected = True
            ts = time.time()
            self.last_frame_ts = ts
            frame = np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3)
            try:
                self.on_frame(ts, frame)
            except Exception:
                log.exception("frame callback failed")

    def _read_audio(self, f) -> None:
        chunk_bytes = int(AUDIO_RATE * AUDIO_CHUNK_S) * 2
        while True:
            buf = _read_exact(f, chunk_bytes)
            if buf is None:
                return
            samples = np.frombuffer(buf, np.int16)
            try:
                self.on_audio(time.time(), samples)
            except Exception:
                log.exception("audio callback failed")

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            msg = redact(line.decode(errors="replace").rstrip())
            if msg:
                log.warning("ffmpeg: %s", msg)


def _read_exact(f, n: int) -> bytes | None:
    """Read exactly n bytes, or return None on EOF."""
    chunks, got = [], 0
    while got < n:
        b = f.read(n - got)
        if not b:
            return None
        chunks.append(b)
        got += len(b)
    return b"".join(chunks) if len(chunks) > 1 else chunks[0]

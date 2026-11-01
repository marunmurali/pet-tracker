import json
import subprocess
import threading
import time

import numpy as np

from app.recorder import Recorder, write_clip
from app.stream import Stream, build_ffmpeg_cmd, redact


def test_redact_hides_credentials():
    assert redact("rtsp://user:p%40ss@10.0.0.2:554/stream2 failed") == "rtsp://***@10.0.0.2:554/stream2 failed"


def test_rtsp_command_uses_tcp_and_one_process():
    cmd = build_ffmpeg_cmd("rtsp://u:p@h:554/stream2", 640, 360, 10, 7, True)
    assert cmd[cmd.index("-rtsp_transport") + 1] == "tcp"
    assert "pipe:1" in cmd and "pipe:7" in cmd
    assert "-re" not in cmd


def test_stream_splits_video_and_audio_from_one_ffmpeg(synthetic_video):
    frames, audio = [], []
    got_enough = threading.Event()

    def on_frame(ts, f):
        frames.append(f)
        if len(frames) >= 15 and sum(len(a) for a in audio) >= 16000:
            got_enough.set()

    s = Stream(str(synthetic_video), 320, 180, 10, on_frame, lambda ts, a: audio.append(a), realtime=False)
    s.start()
    try:
        assert got_enough.wait(20), f"frames={len(frames)} audio={sum(len(a) for a in audio)}"
    finally:
        s.stop()
    assert frames[0].shape == (180, 320, 3)
    pcm = np.concatenate(audio)
    assert pcm.dtype == np.int16 and np.abs(pcm).max() > 1000  # the tone came through
    assert not s.running


def test_stop_kills_ffmpeg(synthetic_video):
    s = Stream(str(synthetic_video), 320, 180, 10, lambda *a: None, lambda *a: None, realtime=True)
    s.start()
    deadline = time.time() + 10
    while not s.connected and time.time() < deadline:
        time.sleep(0.1)
    proc = s._proc
    assert proc is not None and proc.poll() is None
    s.stop()
    assert proc.poll() is not None, "ffmpeg must exit when the stream is switched off"


def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,codec_type",
                          "-show_entries", "format=duration", "-of", "json", str(path)],
                         capture_output=True, check=True, text=True).stdout
    return json.loads(out)


def _fake_frame(i):
    f = np.zeros((180, 320, 3), np.uint8)
    f[:, (i * 10) % 320:] = 200
    return f


def test_write_clip_h264_aac(tmp_path):
    import cv2

    frames = [(100 + i / 10, cv2.imencode(".jpg", _fake_frame(i))[1].tobytes()) for i in range(30)]
    t = np.arange(1600) / 16000
    tone = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
    audio = [(100 + (i + 1) / 10, tone) for i in range(30)]
    out = tmp_path / "clip.mp4"
    assert write_clip(out, frames, audio)
    info = _probe(out)
    codecs = {s["codec_type"]: s["codec_name"] for s in info["streams"]}
    assert codecs == {"video": "h264", "audio": "aac"}
    assert 2.5 < float(info["format"]["duration"]) < 3.5


def test_recorder_includes_pre_and_post(tmp_path):
    rec = Recorder(tmp_path, pre_s=1.0, post_s=1.0)
    done = threading.Event()
    result = {}

    def on_done(p):
        result["path"] = p
        done.set()

    t0 = 1000.0
    for i in range(30):  # 3 s of history at 10 fps
        rec.add_frame(t0 + i / 10, _fake_frame(i))
        rec.add_audio(t0 + i / 10, np.zeros(1600, np.int16))
    rec.trigger(t0 + 2.9, "test", on_done)
    for i in range(30, 45):  # 1.5 s more; the clip should end after 1 s of it
        rec.add_frame(t0 + i / 10, _fake_frame(i))
        rec.add_audio(t0 + i / 10, np.zeros(1600, np.int16))
    assert done.wait(30)
    info = _probe(result["path"])
    assert 1.6 < float(info["format"]["duration"]) < 2.4
    assert not rec.busy

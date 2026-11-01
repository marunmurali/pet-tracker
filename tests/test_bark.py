import subprocess

import numpy as np
import pytest

from app.bark import BARK_LABELS, BarkDetector, load_label_indices
from conftest import MODELS, SAMPLES, have_models

pytestmark = pytest.mark.skipif(not have_models("yamnet.tflite", "yamnet_class_map.csv"),
                                reason="run scripts/setup_models.sh")


def make():
    return BarkDetector(MODELS / "yamnet.tflite", MODELS / "yamnet_class_map.csv", threshold=0.5)


def test_labels_looked_up_by_name():
    idx = load_label_indices(MODELS / "yamnet_class_map.csv")
    assert len(idx) == len(BARK_LABELS) == len(set(idx))


def test_window_size_from_interpreter():
    assert make().window == 15600


def test_silence_is_not_a_bark():
    b = make()
    results = b.push(np.zeros(16000 * 3, np.int16))
    assert len(results) >= 3
    assert not any(ev for _, ev in results)
    assert max(s for s, _ in results) < 0.1


def test_hop_produces_about_two_windows_per_second():
    b = make()
    n = 0
    for _ in range(50):  # 5 s in 100 ms chunks, like the stream delivers it
        n += len(b.push(np.zeros(1600, np.int16)))
    assert 8 <= n <= 10


def test_needs_two_consecutive_hits_and_fires_once_per_episode(monkeypatch):
    b = make()
    scores = iter([0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.2, 0.8, 0.8])
    monkeypatch.setattr(b, "score", lambda w: next(scores))
    events = []
    for _ in range(9):
        events += [ev for _, ev in b.push(np.zeros(8000, np.int16))]
    # one window becomes available only after 15600 samples; feed until 9 scores were used
    while len(events) < 9:
        events += [ev for _, ev in b.push(np.zeros(8000, np.int16))]
    assert events == [False, False, False, True, False, False, False, False, True]


BARK_SAMPLES = sorted(SAMPLES.glob("bark*.wav")) + sorted(SAMPLES.glob("bark*.mp4")) if SAMPLES.exists() else []


@pytest.mark.skipif(not BARK_SAMPLES, reason="put bark*.wav / bark*.mp4 recordings in data/samples/")
@pytest.mark.parametrize("path", BARK_SAMPLES, ids=lambda p: p.name)
def test_detects_bark_in_samples(path):
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                          "-f", "s16le", "pipe:1"], capture_output=True, check=True).stdout
    b = make()
    events = [ev for _, ev in b.push(np.frombuffer(pcm, np.int16))]
    assert any(events)

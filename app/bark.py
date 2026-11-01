"""YAMNet (TFLite) bark detector over a sliding 0.975 s window with ~0.5 s hop."""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter

log = logging.getLogger(__name__)

BARK_LABELS = ("Bark", "Dog", "Bow-wow", "Yip", "Howl", "Growling")
SAMPLE_RATE = 16000


def load_label_indices(class_map_csv: Path, names=BARK_LABELS) -> list[int]:
    with open(class_map_csv, newline="") as f:
        by_name = {row["display_name"]: int(row["index"]) for row in csv.DictReader(f)}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f"labels not in {class_map_csv.name}: {missing}")
    return [by_name[n] for n in names]


class BarkDetector:
    """Feed int16 audio with `push()`; it returns a list of (score, is_bark_event) per evaluated window.

    A bark event fires when `consecutive` windows in a row score >= threshold, and re-arms only after
    a window below the threshold, so one barking fit produces one event per episode.
    """

    def __init__(self, model_path: Path, class_map_csv: Path, threshold: float = 0.5,
                 hop_s: float = 0.5, consecutive: int = 2, threads: int = 1):
        self.interpreter = Interpreter(model_path=str(model_path), num_threads=threads)
        self.interpreter.allocate_tensors()
        inp = self.interpreter.get_input_details()[0]
        self._input_index = inp["index"]
        self._input_shape = tuple(int(d) for d in inp["shape"])
        self.window = int(np.prod(self._input_shape))  # 15600 for the standard export
        self._output_index = self.interpreter.get_output_details()[0]["index"]
        self.label_indices = load_label_indices(class_map_csv)
        self.threshold = threshold
        self.hop = int(hop_s * SAMPLE_RATE)
        self.consecutive = consecutive
        self._buf = np.zeros(0, np.float32)
        self._since_last = 0
        self._hits = 0
        self._armed = True
        self.last_score = 0.0

    def reset(self) -> None:
        self._buf = np.zeros(0, np.float32)
        self._since_last = 0
        self._hits = 0
        self._armed = True

    def score(self, window: np.ndarray) -> float:
        x = window.astype(np.float32).reshape(self._input_shape)
        self.interpreter.set_tensor(self._input_index, x)
        self.interpreter.invoke()
        scores = self.interpreter.get_tensor(self._output_index)
        scores = scores.reshape(-1, scores.shape[-1]).mean(axis=0)  # average if several frames
        return float(scores[self.label_indices].max())

    def push(self, samples_int16: np.ndarray) -> list[tuple[float, bool]]:
        x = samples_int16.astype(np.float32) / 32768.0
        self._buf = np.concatenate([self._buf, x])[-(self.window + self.hop * 4):]
        self._since_last += len(x)
        results = []
        while len(self._buf) >= self.window and self._since_last >= self.hop:
            # evaluate the window ending `since_last - hop` samples before the newest sample
            end = len(self._buf) - (self._since_last - self.hop)
            if end < self.window:
                self._since_last -= self.hop
                continue
            s = self.score(self._buf[end - self.window:end])
            self._since_last -= self.hop
            self.last_score = s
            results.append((s, self._update(s)))
        return results

    def _update(self, s: float) -> bool:
        if s >= self.threshold:
            self._hits += 1
            if self._armed and self._hits >= self.consecutive:
                self._armed = False
                return True
        else:
            self._hits = 0
            self._armed = True
        return False

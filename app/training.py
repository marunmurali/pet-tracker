"""Collects pictures of the dog with YOLO-format box labels, for fine-tuning the detector on him later.

Layout (gitignored, under data/snapshots/training/):
    images/20261101-164823-412_yolo+lock.jpg
    labels/20261101-164823-412_yolo+lock.txt   "0 cx cy w h" (normalised), empty file = no dog
All boxes come from the detector (no hand-drawn boxes, so the labels share one style). The suffix records
which: yolo (confident detection) or yolo+lock (weaker detection confirmed by the lock).
Uncertain boxes (visual tracker only, or a weak detection) go to training/review/ in the same layout:
they are the hard cases worth having, but check them by eye before training on them.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .detector import Detection
from .tracker import iou

REVIEW_BELOW_CONF = 0.3  # yolo+lock boxes under this detector confidence go to review/


class TrainingCollector:
    def __init__(self, root: Path, interval_s: float = 10.0, min_change_iou: float = 0.7, max_still_s: float = 120.0):
        self.root = root
        self.images, self.labels = root / "images", root / "labels"
        self.interval_s = interval_s
        self.min_change_iou = min_change_iou  # skip near-identical shots of a dog that hasn't moved...
        self.max_still_s = max_still_s  # ...but still take one this often
        self._last_t = float("-inf")
        self._last_box: Optional[Detection] = None

    @staticmethod
    def certain(box: Optional[Detection], source: str) -> bool:
        if source == "yolo":
            return True
        return source == "yolo+lock" and box is not None and box.conf >= REVIEW_BELOW_CONF

    def save(self, ts: float, frame: np.ndarray, box: Optional[Detection], source: str) -> Path:
        base = self.root if self.certain(box, source) else self.root / "review"
        images, labels = base / "images", base / "labels"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        stem = time.strftime("%Y%m%d-%H%M%S", time.localtime(ts)) + f"-{int(ts * 1000) % 1000:03d}_{source}"
        cv2.imwrite(str(images / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        line = ""
        if box is not None:
            h, w = frame.shape[:2]
            cx, cy = (box.x1 + box.x2) / 2 / w, (box.y1 + box.y2) / 2 / h
            line = f"0 {cx:.6f} {cy:.6f} {(box.x2 - box.x1) / w:.6f} {(box.y2 - box.y1) / h:.6f}\n"
        (labels / f"{stem}.txt").write_text(line)
        self._last_t, self._last_box = ts, box
        return images / f"{stem}.jpg"

    def maybe_save(self, ts: float, frame: np.ndarray, box: Optional[Detection], source: str) -> Optional[Path]:
        """Rate-limited capture while the dog is followed; skips shots that look like the previous one."""
        if box is None or ts - self._last_t < self.interval_s:
            return None
        moved = self._last_box is None or iou(box, self._last_box) < self.min_change_iou
        if not moved and ts - self._last_t < self.max_still_s:
            return None
        return self.save(ts, frame, box, source)

    def count(self) -> int:
        return len(list(self.images.glob("*.jpg"))) if self.images.exists() else 0

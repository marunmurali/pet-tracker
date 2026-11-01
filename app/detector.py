"""YOLO11n (ONNX, CPU) dog detector.

Works with the stock COCO model (keeps only class 16, "dog") and with a single-class model
fine-tuned on your own dog by scripts/train_dog.sh (class 0).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

DOG_CLASS = 16


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to size x size. Returns (image, scale, (pad_x, pad_y))."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return out, scale, (pad_x, pad_y)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression. boxes: (N, 4) as x1, y1, x2, y2."""
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou < iou_threshold]
    return keep


def decode(output: np.ndarray, conf_threshold: float, scale: float, pad: tuple[int, int],
           frame_shape: tuple[int, int], iou_threshold: float = 0.45) -> list[Detection]:
    """YOLOv8/11 head output (1, 4 + classes, N) -> dog detections in frame pixel coordinates."""
    pred = output[0]
    dog_class = DOG_CLASS if pred.shape[0] == 4 + 80 else 0  # COCO model, or our single-class model
    scores = pred[4 + dog_class]
    mask = scores >= conf_threshold
    if not mask.any():
        return []
    cx, cy, w, h = pred[0][mask], pred[1][mask], pred[2][mask], pred[3][mask]
    scores = scores[mask]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / scale
    fh, fw = frame_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, fw)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, fh)
    keep = nms(boxes, scores, iou_threshold)
    return [Detection(*map(float, boxes[i]), float(scores[i])) for i in keep]


class DogDetector:
    def __init__(self, model_path: Path, imgsz: int = 320, conf_threshold: float = 0.45, threads: int = 2):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        # idle threads busy-wait by default: measured 89% vs 59% of a core at 6 fps for ~4 ms less latency
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.imgsz = shape[2] if isinstance(shape[2], int) else imgsz  # a fixed export size wins
        self.conf_threshold = conf_threshold

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        img, scale, pad = letterbox(frame_bgr, self.imgsz)
        blob = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.session.run(None, {self.input_name: np.ascontiguousarray(blob)})[0]
        return decode(out, self.conf_threshold, scale, pad, frame_bgr.shape[:2])

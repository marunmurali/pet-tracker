from pathlib import Path

import cv2
import numpy as np
import pytest

from app.detector import DOG_CLASS, DogDetector, decode, letterbox, nms
from conftest import MODELS, SAMPLES, have_models


def test_letterbox_keeps_aspect_and_pads():
    img = np.zeros((360, 640, 3), np.uint8)
    out, scale, (px, py) = letterbox(img, 320)
    assert out.shape == (320, 320, 3)
    assert scale == 0.5 and px == 0 and py == 70


def test_nms_drops_overlaps():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    assert nms(boxes, scores, 0.45) == [0, 2]


def test_decode_keeps_only_dogs_and_maps_back_to_frame():
    n = 3
    out = np.zeros((1, 84, n), np.float32)
    # anchor 0: a dog at letterbox centre (160, 160), 40x20
    out[0, :4, 0] = [160, 160, 40, 20]
    out[0, 4 + DOG_CLASS, 0] = 0.9
    # anchor 1: a confident person (class 0) - must be ignored
    out[0, :4, 1] = [100, 100, 40, 40]
    out[0, 4 + 0, 1] = 0.95
    # anchor 2: a weak dog
    out[0, :4, 2] = [200, 200, 10, 10]
    out[0, 4 + DOG_CLASS, 2] = 0.1
    dets = decode(out, 0.45, scale=0.5, pad=(0, 70), frame_shape=(360, 640))
    assert len(dets) == 1
    d = dets[0]
    assert d.center == pytest.approx((320, 180))
    assert (d.x2 - d.x1, d.y2 - d.y1) == pytest.approx((80, 40))


@pytest.mark.skipif(not have_models("yolo11n.onnx"), reason="run scripts/setup_models.sh")
def test_model_runs_on_empty_frame():
    det = DogDetector(MODELS / "yolo11n.onnx")
    assert det.detect(np.zeros((360, 640, 3), np.uint8)) == []


DOG_IMAGES = sorted(SAMPLES.glob("dog*.jpg")) if SAMPLES.exists() else []


@pytest.mark.skipif(not (DOG_IMAGES and have_models("yolo11n.onnx")),
                    reason="put dog*.jpg images in data/samples/")
@pytest.mark.parametrize("path", DOG_IMAGES, ids=lambda p: Path(p).name)
def test_finds_dog_in_samples(path):
    det = DogDetector(MODELS / "yolo11n.onnx")
    assert det.detect(cv2.imread(str(path)))

#!/usr/bin/env bash
# Fetches the two models the tracker needs into ./models
#   - yolo11n.onnx         dog detector (COCO class 16 = "dog"), exported from Ultralytics
#   - yamnet.tflite        audio classifier (AudioSet, 521 classes incl. "Bark")
#   - yamnet_class_map.csv labels for YAMNet
#   - vittrack.onnx        OpenCV ViT single-object tracker (the dashboard's Locate / lock)
# Re-running skips anything already present. The YOLO export size comes from YOLO_IMGSZ (environment,
# else .env, else 320). To change it: set YOLO_IMGSZ in .env, delete models/yolo11n.onnx, re-run.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models
if [ -z "${YOLO_IMGSZ:-}" ] && [ -f .env ]; then
  YOLO_IMGSZ="$(sed -n 's/^YOLO_IMGSZ=\([0-9]*\).*/\1/p' .env | tail -n 1)"
fi
IMGSZ="${YOLO_IMGSZ:-320}"
ULTRALYTICS_VERSION=8.4.197  # pinned so the export is reproducible

if [ ! -f models/yamnet.tflite ]; then
  echo "==> Downloading YAMNet"
  curl -fL -o models/yamnet.tflite \
    https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite
fi

if [ ! -f models/yamnet_class_map.csv ]; then
  curl -fL -o models/yamnet_class_map.csv \
    https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv
fi

if [ ! -f models/vittrack.onnx ]; then
  echo "==> Downloading ViT tracker (follows the dog after you locate him once)"
  curl -fL -o models/vittrack.onnx \
    https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx
fi

if [ ! -f models/yolo11n.onnx ]; then
  echo "==> Exporting YOLO11n to ONNX (imgsz=${IMGSZ}); torch is installed in a throwaway venv"
  VENV="$(mktemp -d)/yolo-export"
  python -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q --index-url https://download.pytorch.org/whl/cpu torch torchvision
  "$VENV/bin/pip" install -q "ultralytics==${ULTRALYTICS_VERSION}" onnx onnxslim
  WORK="$(mktemp -d)"
  ( cd "$WORK" && "$VENV/bin/python" - <<PY
from ultralytics import YOLO
YOLO("yolo11n.pt").export(format="onnx", imgsz=${IMGSZ}, opset=12, simplify=True)
PY
  )
  mv "$WORK/yolo11n.onnx" models/yolo11n.onnx
  rm -rf "$WORK" "$(dirname "$VENV")"
fi

# The app reads the size from the model itself; warn if it differs from what .env asks for.
HAVE="$(python3 -c 'import onnxruntime as o; print(o.InferenceSession("models/yolo11n.onnx").get_inputs()[0].shape[2])' 2>/dev/null || true)"
if [ -n "$HAVE" ] && [ "$HAVE" != "$IMGSZ" ]; then
  echo "!! models/yolo11n.onnx is ${HAVE}px but YOLO_IMGSZ=${IMGSZ}. Delete it and re-run to re-export."
fi

echo "==> Models ready:"
ls -lh models

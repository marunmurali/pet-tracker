#!/usr/bin/env python3
"""Run the dog detector and bark detector over a recorded file, offline (no camera needed).

    python scripts/analyze.py data/samples/living-room.mp4 [--fps 5] [--save-frames out_dir]

Prints a per-second line (dog seen? confidence, bark score) and the bark events, which is the
quickest way to tune DOG_CONF_THRESHOLD and BARK_THRESHOLD against real footage.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.bark import BarkDetector  # noqa: E402
from app.config import load_config  # noqa: E402
from app.detector import DogDetector  # noqa: E402


def read_audio(path: Path) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                          "-f", "s16le", "pipe:1"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16)


def read_frames(path: Path, fps: float, w: int, h: int):
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={w}:{h}",
                             "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"], stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    proc.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=Path)
    ap.add_argument("--fps", type=float, default=5)
    ap.add_argument("--save-frames", type=Path, help="write annotated frames with detections here")
    args = ap.parse_args()
    cfg = load_config()

    det = DogDetector(cfg.models_dir / cfg.yolo_model, cfg.yolo_imgsz, cfg.dog_conf_threshold)
    per_sec: dict[int, list[float]] = {}
    for i, frame in enumerate(read_frames(args.file, args.fps, cfg.frame_width, cfg.frame_height)):
        dets = det.detect(frame)
        sec = int(i / args.fps)
        per_sec.setdefault(sec, []).append(max((d.conf for d in dets), default=0.0))
        if args.save_frames and dets:
            args.save_frames.mkdir(parents=True, exist_ok=True)
            frame = frame.copy()
            for d in dets:
                cv2.rectangle(frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 200, 255), 2)
            cv2.imwrite(str(args.save_frames / f"{i:05d}.jpg"), frame)

    bark = BarkDetector(cfg.models_dir / "yamnet.tflite", cfg.models_dir / "yamnet_class_map.csv",
                        cfg.bark_threshold)
    audio = read_audio(args.file)
    scores: dict[int, float] = {}
    events: list[float] = []
    pos = 0
    for start in range(0, len(audio), 1600):  # 100 ms chunks, like the live stream
        chunk = audio[start:start + 1600]
        pos += len(chunk)
        for s, ev in bark.push(chunk):
            t = pos / 16000
            scores[int(t)] = max(scores.get(int(t), 0.0), s)
            if ev:
                events.append(t)

    print(f"{'sec':>4}  {'dog':>4}  {'conf':>5}  {'bark':>5}")
    for sec in range(max([*per_sec, *scores, 0]) + 1):
        confs = per_sec.get(sec, [0.0])
        seen = sum(c > 0 for c in confs)
        b = scores.get(sec, 0.0)
        print(f"{sec:>4}  {seen:>2}/{len(confs):<1}  {max(confs):5.2f}  {b:5.2f}{'  BARK' if b >= cfg.bark_threshold else ''}")
    frames = [c for v in per_sec.values() for c in v]
    print(f"\ndog in {sum(c > 0 for c in frames)}/{len(frames)} frames "
          f"(threshold {cfg.dog_conf_threshold}); bark events at: "
          f"{', '.join(f'{t:.1f}s' for t in events) or 'none'} (threshold {cfg.bark_threshold})")


if __name__ == "__main__":
    main()

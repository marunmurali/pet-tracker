#!/usr/bin/env python3
"""Replay a recording through the detector + lock and write it with the dashboard's overlay.

    python scripts/render_demo.py data/samples/home.mkv out.mp4 [--start 12 --duration 10]
    python scripts/render_demo.py data/samples/home.mkv docs/demo.gif --start 12 --duration 8 --width 480

Output format follows the extension: .mp4 (H.264) or .gif (palette-optimised, for the README).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_config  # noqa: E402
from app.detector import DogDetector  # noqa: E402
from app.tracker import DogLock  # noqa: E402
from scripts.analyze import read_frames  # noqa: E402

GREEN, ORANGE = (80, 220, 80), (0, 200, 255)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--start", type=float, default=0, help="seconds into the recording")
    ap.add_argument("--duration", type=float, default=0, help="seconds to keep (0 = to the end)")
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()
    cfg = load_config()

    det = DogDetector(cfg.models_dir / cfg.yolo_model, cfg.yolo_imgsz, 0.10)
    lock = DogLock(cfg.models_dir / "vittrack.onnx", 0.10, cfg.dog_conf_threshold)
    with tempfile.TemporaryDirectory() as td:
        n = 0
        for i, frame in enumerate(read_frames(args.src, args.fps, cfg.frame_width, cfg.frame_height)):
            t = i / args.fps
            box = lock.update(t, frame, det.detect(frame))  # run from the start so the lock is warm
            if t < args.start or (args.duration and t > args.start + args.duration):
                continue
            img = frame.copy()
            if box is not None:
                color = GREEN if lock.locked else ORANGE
                label = f"{cfg.dog_name} ({lock.source})" if lock.locked else f"dog {box.conf:.2f}"
                cv2.rectangle(img, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, 2)
                cv2.putText(img, label, (int(box.x1) + 3, max(14, int(box.y1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            cv2.imwrite(f"{td}/{n:05d}.png", img)
            n += 1
        if not n:
            raise SystemExit("no frames in that range")
        frames = ["-framerate", str(args.fps), "-i", f"{td}/%05d.png"]
        scale = f"scale={args.width}:-2:flags=lanczos"
        if args.out.suffix.lower() == ".gif":
            vf = f"{scale},split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
            cmd = ["ffmpeg", "-v", "error", "-y", *frames, "-vf", vf, "-loop", "0", str(args.out)]
        else:
            cmd = ["ffmpeg", "-v", "error", "-y", *frames, "-vf", scale, "-c:v", "libx264", "-crf", "26",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out)]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True)
    print(f"wrote {args.out} ({n} frames, {args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Runs once after the container is created.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Preparing local files"
mkdir -p data/clips data/snapshots data/samples models
[ -f .env ] || { cp .env.example .env; echo "Created .env from .env.example - fill it in."; }

echo "==> Installing test tools"
pip install --user -q -r requirements-dev.txt

echo "==> Downloading / exporting models (safe to re-run: bash scripts/setup_models.sh)"
bash scripts/setup_models.sh || echo "!! Model setup failed - rerun 'bash scripts/setup_models.sh' once online."

echo
echo "Container ready. Next: fill in .env, then run 'bash scripts/check_camera.sh'."

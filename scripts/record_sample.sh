#!/usr/bin/env bash
# Records footage (video + audio, stream copied, no re-encode) from the camera to data/samples/,
# for offline testing:  python scripts/analyze.py data/samples/<file>.mkv
#   bash scripts/record_sample.sh [minutes=3] [name=sample]
# Stop the app first (or /off): the Tapo allows only a couple of RTSP sessions.
set -euo pipefail

cd "$(dirname "$0")/.."
set -a; source .env; set +a

: "${CAM_HOST:?Set CAM_HOST in .env}" "${CAM_USER:?Set CAM_USER in .env}" "${CAM_PASS:?Set CAM_PASS in .env}"
MINUTES="${1:-3}"
NAME="${2:-sample}"
STREAM="${CAM_STREAM:-stream2}"

enc() { python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1],safe=""))' "$1"; }
URL="rtsp://$(enc "$CAM_USER"):$(enc "$CAM_PASS")@${CAM_HOST}:554/${STREAM}"
mkdir -p data/samples
OUT="data/samples/${NAME}-$(date +%Y%m%d-%H%M%S).mkv"

echo "==> Recording ${MINUTES} min of ${STREAM} to ${OUT} (Ctrl+C stops early)"
ffmpeg -v error -stats -rtsp_transport tcp -i "$URL" -map 0 -c copy -t "$((MINUTES * 60))" "$OUT"
echo "Saved ${OUT}"

#!/usr/bin/env bash
# Smoke test: can the container reach the Tapo camera's video, audio and ONVIF service?
# Writes data/snapshots/check.jpg and data/snapshots/check.wav on success.
set -uo pipefail

cd "$(dirname "$0")/.."
set -a; source .env; set +a

: "${CAM_HOST:?Set CAM_HOST in .env}" "${CAM_USER:?Set CAM_USER in .env}" "${CAM_PASS:?Set CAM_PASS in .env}"
STREAM="${CAM_STREAM:-stream2}"
ONVIF_PORT="${ONVIF_PORT:-2020}"

# URL-encode credentials so passwords with @, :, / etc. still work.
enc() { python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1],safe=""))' "$1"; }
URL="rtsp://$(enc "$CAM_USER"):$(enc "$CAM_PASS")@${CAM_HOST}:554/${STREAM}"
mkdir -p data/snapshots
ok=0

echo "==> 1/3 Stream info (${STREAM})"
if ffprobe -v error -rtsp_transport tcp -show_entries stream=codec_type,codec_name,width,height,sample_rate \
     -of compact "$URL"; then ok=$((ok+1)); else echo "!! RTSP failed: check IP, camera account, and that the camera is on the same network"; fi

echo "==> 2/3 Grabbing a frame and 5 s of audio"
if ffmpeg -v error -y -rtsp_transport tcp -i "$URL" -frames:v 1 data/snapshots/check.jpg \
   && ffmpeg -v error -y -rtsp_transport tcp -i "$URL" -vn -ac 1 -ar 16000 -t 5 data/snapshots/check.wav; then
  ok=$((ok+1)); echo "   saved data/snapshots/check.jpg and check.wav"
else
  echo "!! Could not capture frame/audio"
fi

echo "==> 3/3 ONVIF service on port ${ONVIF_PORT} (needed for pan/tilt)"
BODY='<?xml version="1.0"?><s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"><s:Body><GetSystemDateAndTime xmlns="http://www.onvif.org/ver10/device/wsdl"/></s:Body></s:Envelope>'
if curl -fsS -m 5 -H 'Content-Type: application/soap+xml' -d "$BODY" \
     "http://${CAM_HOST}:${ONVIF_PORT}/onvif/device_service" | grep -q 'UTCDateTime'; then
  ok=$((ok+1)); echo "   ONVIF reachable"
else
  echo "!! ONVIF not reachable - pan/tilt will need the pytapo fallback"
fi

echo
echo "${ok}/3 checks passed."
[ "$ok" -eq 3 ]

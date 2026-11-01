# pet-tracker — build spec

A home service that watches a dog through a TP-Link **Tapo C200/C210/C211** pan/tilt camera,
follows him with pan/tilt, logs what he does, and sends Telegram alerts (with a clip) when he barks.
Runs in this devcontainer now; later as an always-on container on a **Raspberry Pi 5 (arm64)**.

## Hard constraints
- Everything must run on CPU and on arm64. No CUDA, no packages without aarch64 wheels.
- Keep load low enough for a Pi 5: detection on `stream2` (640x360) at ~5-10 fps, YOLO at 320 px.
- Config comes only from `.env` (see `.env.example`). Secrets never logged or committed.
- **Off means off:** when the feature is disabled, close the RTSP connection entirely
  (no video, no audio, no recording), not just pause processing.
- The dashboard is LAN-only and requires `DASHBOARD_TOKEN`. For remote access, document Tailscale; never port-forward.
- Telegram bot only obeys `TELEGRAM_CHAT_ID`.

## Camera facts
- RTSP: `rtsp://USER:PASS@HOST:554/stream1|stream2` using the *camera account* (URL-encode credentials).
  Audio track is G.711 in both streams. Use TCP transport.
- Tapo allows only a couple of concurrent RTSP sessions. Open **one** ffmpeg process and split it:
  raw BGR frames on `pipe:1` and 16 kHz mono s16le audio on `pipe:3` (via `pass_fds`), each read on its own thread.
  Reconnect with backoff when the stream drops.
- Pan/tilt: ONVIF on port 2020 (`onvif-zeep`, ContinuousMove + Stop in short pulses, or RelativeMove if supported).
  Fallback backend: `pytapo` `moveMotor(x, y)` using the TP-Link cloud password. Make the backend selectable.

## Models (in `models/`, created by `scripts/setup_models.sh`)
- `yolo11n.onnx` via `onnxruntime`; keep only COCO class 16 (dog). Letterbox input, NMS in numpy.
- `yamnet.tflite` via `ai_edge_litert`: 0.975 s windows (15600 samples @ 16 kHz), hop ~0.5 s.
  Read the input shape from the interpreter. Bark score = max over labels looked up **by name** from
  `yamnet_class_map.csv`: Bark, Dog, Bow-wow, Yip, Howl, Growling. Require 2 consecutive hits.

## Features
1. **Detection + tracking** — dog bbox → simple deadband P-controller on the bbox centre → pan/tilt pulse,
   then ignore motion estimates for a settle time (~0.8 s) while the image is moving. If the dog is lost
   for N seconds, stop moving (optional: return to an ONVIF preset).
2. **Activity log** — states `absent | resting | active` from bbox presence and centroid motion
   (hysteresis, minimum durations). Store state intervals and events in SQLite (`data/pet-tracker.db`).
   Optional named zones (bed, bowl, door) valid only when the camera is at its home position.
3. **Bark alerts** — on a bark event: save a clip (CLIP_PRE_S before from a ring buffer + CLIP_POST_S after,
   H.264 + AAC via ffmpeg) to `data/clips/`, send a Telegram message + video, respect BARK_ALERT_COOLDOWN_S
   (count suppressed barks and mention them in the next alert).
4. **Daily summary** at DAILY_SUMMARY_TIME: time per state, number of barks, longest rest, a few snapshots.
5. **Dashboard** (FastAPI, port 8000): live MJPEG with bbox overlay, current state, on/off toggle,
   tracking toggle, today's timeline, bark list with clip playback. Mobile-friendly (viewed on iPhone).
6. **On/off control** from two places, one shared state persisted in SQLite:
   Telegram (`/on /off /status /snap /clip /summary /help`) and the dashboard toggle,
   backed by an HTTP API (`GET /api/state`, `POST /api/enabled`, token in header).
   No ESP32 or other hardware button: PC first, then Raspberry Pi.

## Suggested layout
```
app/
  main.py         orchestrator, threads, graceful shutdown
  config.py       .env -> dataclass
  stream.py       single ffmpeg RTSP reader (video + audio), reconnects
  detector.py     YOLO ONNX dog detector
  bark.py         YAMNet bark detector
  ptz.py          ONVIF / pytapo backends behind one interface
  tracker.py      controller + activity state machine
  recorder.py     ring buffers, clip writer
  store.py        SQLite (events, intervals, settings)
  telegram_bot.py long-polling bot (requests only)
  web.py          FastAPI dashboard + API; templates/ static/
docker-compose.yml    always-on deployment reusing .devcontainer/Dockerfile, restart: unless-stopped
tests/                offline tests using recorded clips in data/ (no camera needed)
```

## Working style
- Build in milestones, each runnable: stream → detection overlay → dashboard → bark + Telegram →
  tracking → activity log + summary → on/off everywhere → compose (PC) → Pi.
- `scripts/check_camera.sh` is the smoke test; run it before debugging app code.
- Record a few minutes of real footage to `data/samples/` early and test against it offline.

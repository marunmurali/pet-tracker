"""Orchestrator: wires the stream, detectors, tracker, recorder, store, Telegram bot and dashboard.

Run with `python -m app.main`.
"""
from __future__ import annotations

import datetime as dt
import logging
import queue
import re
import shutil
import signal
import threading
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from . import summary
from .bark import BarkDetector
from .config import Config, load_config
from .detector import Detection, DogDetector
from .ptz import PTZ
from .recorder import Recorder, clip_name
from .store import Store
from .stream import Stream
from .telegram_bot import TelegramBot
from .tracker import ActivityTracker, DogLock, PanTiltController
from .training import TrainingCollector

log = logging.getLogger("pet-tracker")

STALE_STREAM_S = 30  # no frames this long while on -> stop counting activity until frames return
RETENTION_DAYS = 30  # clips and dated snapshot folders older than this are deleted (never training/)
LOW_CONF = 0.10  # YOLO threshold used to confirm a locked dog; unlocked, DOG_CONF_THRESHOLD applies


class App:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tz = ZoneInfo(cfg.tz)
        for d in (cfg.clips_dir, cfg.snapshots_dir):
            d.mkdir(parents=True, exist_ok=True)
        legacy_db = cfg.data_dir / "dogcam.db"  # the project's old name; keep the existing log
        if legacy_db.exists() and not cfg.db_path.exists():
            for suffix in ("", "-wal", "-shm"):
                old = legacy_db.with_name(legacy_db.name + suffix)
                if old.exists():
                    old.rename(cfg.db_path.with_name(cfg.db_path.name + suffix))
        self.store = Store(cfg.db_path)
        self.detector = DogDetector(cfg.models_dir / cfg.yolo_model, cfg.yolo_imgsz,
                                    min(LOW_CONF, cfg.dog_conf_threshold))
        self.lock = DogLock(cfg.models_dir / "vittrack.onnx", LOW_CONF, cfg.dog_conf_threshold)
        self.training = (TrainingCollector(cfg.snapshots_dir / "training", cfg.training_interval_s)
                         if cfg.training_capture else None)
        self.bark = BarkDetector(cfg.models_dir / "yamnet.tflite", cfg.models_dir / "yamnet_class_map.csv",
                                 cfg.bark_threshold)
        self.ptz = PTZ(cfg.ptz_backend, cfg.cam_host, cfg.onvif_port, cfg.cam_user, cfg.cam_pass,
                       cfg.tapo_cloud_password, cfg.ptz_invert_pan, cfg.ptz_invert_tilt)
        self.controller = PanTiltController(self.ptz, cfg.frame_width, cfg.frame_height,
                                            lost_timeout_s=cfg.lost_timeout_s, home_preset=cfg.ptz_home_preset,
                                            sweep=cfg.sweep_on_lost, sweep_max_s=cfg.sweep_max_s,
                                            sweep_min_gap_s=cfg.sweep_min_gap_s)
        self.activity = ActivityTracker(on_change=self._on_state_change, zones=cfg.zones, on_zone=self._on_zone)
        self.recorder = Recorder(cfg.clips_dir, cfg.clip_pre_s, cfg.clip_post_s)
        self.bot = TelegramBot(cfg.telegram_bot_token, cfg.telegram_chat_id)
        self.stream = Stream(cfg.source, cfg.frame_width, cfg.frame_height, cfg.stream_fps,
                             self._on_frame, self._on_audio)

        self.enabled: bool = self.store.get("enabled", True)
        self.tracking: bool = self.store.get("tracking", cfg.tracking_enabled)

        self._stop = threading.Event()
        self._power_lock = threading.Lock()
        self._frame_cond = threading.Condition()
        self._frame: tuple[float, np.ndarray] | None = None
        self._dets: list[Detection] = []  # confident detections, for the overlay
        self._target: Detection | None = None  # the box we follow (from YOLO and/or the lock)
        self._det_seq = 0
        self._lock_mu = threading.Lock()  # DogLock is used by the detect thread and by on/off requests
        self._jpeg_cache: tuple[tuple, bytes] | None = None
        self._audio_q: queue.Queue = queue.Queue(maxsize=50)
        self._interval_open = False
        self._last_snapshot = 0.0
        # bark alerts
        self._last_alert = 0.0
        self._suppressed = 0
        self._clip_lock = threading.Lock()
        self._clip: dict | None = None  # {"name", "end", "events": [ids]} for the clip being recorded
        self._threads: list[threading.Thread] = []

    # --- lifecycle ---------------------------------------------------------------------------------
    def start(self) -> None:
        for name, target in (("detect", self._detect_loop), ("bark", self._bark_loop),
                             ("scheduler", self._scheduler_loop)):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        self.bot.start(self.handle_command)
        if self.enabled:
            self.stream.start()
        log.info("pet-tracker started (enabled=%s, tracking=%s, ptz=%s)", self.enabled, self.tracking,
                 self.cfg.ptz_backend)

    def stop(self) -> None:
        log.info("shutting down")
        self._stop.set()
        self.bot.stop()
        self.stream.stop()
        self.recorder.abort()
        self.recorder.wait()
        self._close_interval(time.time())
        with self._frame_cond:
            self._frame_cond.notify_all()
        for t in self._threads:
            t.join(timeout=5)
        self.ptz.close()
        self.store.close()

    # --- shared on/off + tracking state ------------------------------------------------------------
    def set_enabled(self, on: bool, source: str) -> None:
        with self._power_lock:
            if on == self.enabled:
                return
            self.enabled = on
            self.store.set("enabled", on)
            self.store.add_event("power", data={"on": on, "source": source})
            log.info("turned %s from %s", "on" if on else "off", source)
            if on:
                self.bark.reset()
                self.stream.start()
            else:
                self.stream.stop()  # closes the RTSP session: no video, no audio, no recording
                self.recorder.abort()
                self.recorder.clear()
                self.controller.stop_sweep(time.time())
                self.ptz.stop()
                self._close_interval(time.time())
                with self._lock_mu:
                    self.lock.unlock()
                with self._frame_cond:
                    self._frame = None
                    self._dets = []
                    self._target = None
        if source != "telegram":
            self.bot.send_message(f"{'🟢 Watching' if on else '🔴 Off'} (switched from {source})")

    def set_tracking(self, on: bool, source: str) -> None:
        if on == self.tracking:
            return
        self.tracking = on
        self.store.set("tracking", on)
        log.info("tracking %s from %s", "on" if on else "off", source)
        if not on:
            self.controller.stop_sweep(time.time())
            self.ptz.stop()

    def status(self) -> dict:
        now = time.time()
        start, end = summary.day_bounds(self.today(), self.tz)
        return {
            "enabled": self.enabled,
            "tracking": self.tracking,
            "connected": self.enabled and self.stream.connected,
            "state": self.activity.state if self._interval_open else None,
            "since": self.activity.since if self._interval_open else None,
            "zone": self.activity.zone,
            "dog_visible": self._target is not None,
            "locked": self.lock.locked,
            "lock_source": self.lock.source,
            "sweeping": self.controller.sweeping,
            "training_images": self.training.count() if self.training else None,
            "barks_today": self.store.count_events(start, end, "bark"),
            "ptz": self.cfg.ptz_backend,
            "ptz_available": self.ptz.available,
            "ptz_moves_today": self.ptz.moves_today,
            "ptz_motor_s_today": round(self.ptz.motor_s_today),
            "dog_name": self.cfg.dog_name,
            "now": now,
        }

    def today(self) -> dt.date:
        return dt.datetime.now(self.tz).date()

    # --- stream callbacks (stream threads) ---------------------------------------------------------
    def _on_frame(self, ts: float, frame: np.ndarray) -> None:
        with self._frame_cond:
            self._frame = (ts, frame)
            self._frame_cond.notify_all()
        self.recorder.add_frame(ts, frame)

    def _on_audio(self, ts: float, samples: np.ndarray) -> None:
        self.recorder.add_audio(ts, samples)
        try:
            self._audio_q.put_nowait((ts, samples))
        except queue.Full:
            pass  # bark worker is behind; dropping audio is better than growing latency

    # --- detection + tracking + activity -----------------------------------------------------------
    def _detect_loop(self) -> None:
        period = 1.0 / self.cfg.detect_fps
        last_ts = 0.0
        while not self._stop.is_set():
            with self._frame_cond:
                self._frame_cond.wait_for(lambda: self._stop.is_set() or
                                          (self._frame is not None and self._frame[0] > last_ts), timeout=1.0)
                item = self._frame
            if self._stop.is_set():
                return
            if item is None or item[0] <= last_ts:
                if self._interval_open and time.time() - last_ts > STALE_STREAM_S:
                    log.warning("no frames for %d s; pausing the activity log", STALE_STREAM_S)
                    self._close_interval(last_ts)
                continue
            ts, frame = item
            last_ts = ts
            started = time.monotonic()
            try:
                self._process(ts, frame)
            except Exception:
                log.exception("detection step failed")
            self._stop.wait(max(0.0, period - (time.monotonic() - started)))

    def _process(self, ts: float, frame: np.ndarray) -> None:
        dets = self.detector.detect(frame)
        if not self.enabled:
            return  # switched off while detecting
        with self._lock_mu:
            was_locked = self.lock.locked
            det = self.lock.update(ts, frame, dets)
            source = self.lock.source
        if was_locked and not self.lock.locked:
            log.info("lost %s for %.0f s; lock released", self.cfg.dog_name, self.lock.lost_after_s)
        if not self._interval_open:
            self.activity.reset()
            self.activity.since = ts
            self._interval_open = True
            self.store.start_interval(self.activity.state, ts)
        if self.tracking and self.ptz.enabled:
            was_sweeping = self.controller.sweeping
            self.controller.update(ts, det, frame)
            if self.controller.sweeping != was_sweeping:
                log.info("%s %s", self.cfg.dog_name,
                         "lost: sweeping to find him" if self.controller.sweeping else
                         ("found" if det is not None else "not found; sweep stopped"))
        moving = self.tracking and self.controller.camera_moving(ts)
        self.activity.update(ts, det, frame.shape[1], frame.shape[0], moving, self.controller.at_home)
        self._dets = [d for d in dets if d.conf >= self.cfg.dog_conf_threshold]
        self._target = det
        self._det_seq += 1
        if self.training:
            self.training.maybe_save(ts, frame, det, source)
        if det is not None and ts - self._last_snapshot >= self.cfg.snapshot_interval_min * 60:
            self._last_snapshot = ts
            self._save_snapshot(ts, frame)

    def _save_snapshot(self, ts: float, frame: np.ndarray) -> None:
        local = dt.datetime.fromtimestamp(ts, self.tz)
        d = self.cfg.snapshots_dir / local.date().isoformat()
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{local:%H%M%S}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    def _on_state_change(self, old: str, new: str, t_start: float) -> None:
        log.info("activity: %s -> %s", old, new)
        if self._interval_open:
            self.store.start_interval(new, t_start)

    def _on_zone(self, zone: str | None, ts: float) -> None:
        log.info("zone: %s", zone or "none")
        self.store.add_event("zone", ts, {"zone": zone})

    def _close_interval(self, ts: float) -> None:
        if self._interval_open:
            self._interval_open = False
            self.store.close_interval(ts)
            self.activity.reset()

    # --- barks + clips -----------------------------------------------------------------------------
    def _bark_loop(self) -> None:
        while not self._stop.is_set():
            try:
                ts, samples = self._audio_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if not self.enabled:
                continue
            try:
                for score, is_event in self.bark.push(samples):
                    if is_event:
                        self._on_bark(ts, score)
            except Exception:
                log.exception("bark detection failed")

    def _on_bark(self, ts: float, score: float) -> None:
        alert = ts - self._last_alert >= self.cfg.bark_alert_cooldown_s
        event_id = self.store.add_event("bark", ts, {"score": round(score, 3), "alerted": alert})
        log.info("bark (score %.2f)%s", score, "" if alert else " - alert suppressed by cooldown")
        self._attach_clip(ts, event_id, alert)
        if alert:
            suppressed, self._suppressed = self._suppressed, 0
            self._last_alert = ts
            extra = f" (+{suppressed} more since the last alert)" if suppressed else ""
            self.bot.send_message(f"🔊 {self.cfg.dog_name} is barking!{extra} Clip coming…")
        else:
            self._suppressed += 1

    def _attach_clip(self, ts: float, event_id: int, send: bool) -> None:
        """Every bark gets a clip; barks during a clip that's still recording share it."""
        with self._clip_lock:
            if self._clip and ts < self._clip["end"]:
                self._clip["events"].append(event_id)
                return
            name = clip_name("bark", ts)
            clip = {"name": name, "end": ts + self.cfg.clip_post_s, "events": [event_id], "send": send}
            self._clip = clip

        def done(path: Path | None) -> None:
            if path is None:
                return
            for eid in clip["events"]:
                self.store.set_event_clip(eid, path.name)
            if clip["send"]:
                when = dt.datetime.fromtimestamp(ts, self.tz).strftime("%H:%M:%S")
                self.bot.send_video(path, f"🔊 {self.cfg.dog_name} barking at {when}")

        self.recorder.trigger(ts, name, done)

    def record_clip(self, on_done) -> bool:
        if not (self.enabled and self.stream.connected):
            return False
        ts = time.time()
        self.recorder.trigger(ts, clip_name("manual", ts), on_done)
        return True

    # --- frames for the dashboard / Telegram -------------------------------------------------------
    def annotated_jpeg(self) -> bytes | None:
        """Latest frame with bbox overlay, JPEG-encoded (cached per frame)."""
        with self._frame_cond:
            item, dets, target = self._frame, list(self._dets), self._target
        if item is None:
            return None
        key = (item[0], self._det_seq)
        cache = self._jpeg_cache
        if cache and cache[0] == key:
            return cache[1]
        img = item[1].copy()
        for d in dets:
            if d is not target:
                cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 200, 255), 1)
        if target is not None:
            color = (80, 220, 80) if self.lock.locked else (0, 200, 255)
            text = f"{self.cfg.dog_name} ({self.lock.source})" if self.lock.locked else f"dog {target.conf:.2f}"
            cv2.rectangle(img, (int(target.x1), int(target.y1)), (int(target.x2), int(target.y2)), color, 2)
            cv2.putText(img, text, (int(target.x1) + 3, max(14, int(target.y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        label = f"{dt.datetime.fromtimestamp(item[0], self.tz):%H:%M:%S}  {self.activity.state}"
        cv2.putText(img, label, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            return None
        self._jpeg_cache = (key, jpg.tobytes())
        return self._jpeg_cache[1]

    # --- daily summary + housekeeping --------------------------------------------------------------
    def send_summary(self, day: dt.date) -> str:
        stats = summary.day_stats(self.store, day, self.tz)
        text = summary.format_summary(stats, self.cfg.dog_name, self.tz)
        snaps = summary.pick_snapshots(self.cfg.snapshots_dir, day)
        if snaps:
            self.bot.send_photos(snaps, text)
        else:
            self.bot.send_message(text)
        return text

    def _scheduler_loop(self) -> None:
        hh, mm = (int(x) for x in self.cfg.daily_summary_time.split(":"))
        while not self._stop.wait(20):
            now = dt.datetime.now(self.tz)
            today = now.date().isoformat()
            if (now.hour, now.minute) >= (hh, mm) and self.store.get("last_summary") != today:
                self.store.set("last_summary", today)
                try:
                    self.send_summary(now.date())
                except Exception:
                    log.exception("daily summary failed")
            if self.store.get("last_cleanup") != today:
                self.store.set("last_cleanup", today)
                self._cleanup(now.date() - dt.timedelta(days=RETENTION_DAYS))

    def _cleanup(self, before: dt.date) -> None:
        cutoff = time.time() - RETENTION_DAYS * 86400
        for f in self.cfg.clips_dir.glob("*.mp4"):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
        for d in self.cfg.snapshots_dir.iterdir():  # only dated folders: training/ is kept
            if d.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.name) and d.name < before.isoformat():
                shutil.rmtree(d, ignore_errors=True)

    # --- Telegram commands -------------------------------------------------------------------------
    def status_text(self) -> str:
        s = self.status()
        if not s["enabled"]:
            return f"🔴 Off. {self.cfg.dog_name}: not watching. /on to start."
        lines = [f"🟢 Watching, tracking {'on' if s['tracking'] else 'off'}"]
        if not s["connected"]:
            lines.append("⚠️ Camera not connected (retrying)")
        if s["state"]:
            mins = int((s["now"] - s["since"]) / 60) if s["since"] else 0
            zone = f" in {s['zone']}" if s["zone"] else ""
            lines.append(f"{self.cfg.dog_name}: {s['state']}{zone} for {mins} min")
        lines.append(f"Barks today: {s['barks_today']}")
        return "\n".join(lines)

    def handle_command(self, cmd: str, arg: str) -> str | None:
        if cmd == "on":
            self.set_enabled(True, "telegram")
            return "🟢 Watching."
        if cmd == "off":
            self.set_enabled(False, "telegram")
            return "🔴 Off. Camera stream closed."
        if cmd == "status":
            return self.status_text()
        if cmd == "snap":
            jpg = self.annotated_jpeg() if self.enabled else None
            if jpg is None:
                return "No picture: the camera is off or not connected."
            self.bot.send_photo(jpg, self.status_text())
            return None
        if cmd == "clip":
            def done(path):
                if path:
                    self.bot.send_video(path, "🎬 Requested clip")
                else:
                    self.bot.send_message("Clip failed.")
            if not self.record_clip(done):
                return "Can't record: the camera is off or not connected."
            return f"🎬 Recording ({self.cfg.clip_pre_s:.0f} s before + {self.cfg.clip_post_s:.0f} s after)…"
        if cmd == "summary":
            self.send_summary(self.today())
            return None
        return "Unknown command. /help"


def _exit_on_sigterm(signum, frame) -> None:
    raise SystemExit(0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("zeep").setLevel(logging.WARNING)
    cfg = load_config()
    if not cfg.cam_source and not (cfg.cam_host and cfg.cam_user and cfg.cam_pass):
        raise SystemExit("Set CAM_HOST, CAM_USER and CAM_PASS in .env (or CAM_SOURCE for a recorded file).")

    import uvicorn

    from .web import create_app

    app = App(cfg)
    app.start()
    # timeout_graceful_shutdown: open MJPEG views never end on their own; don't wait for them forever
    server = uvicorn.Server(uvicorn.Config(create_app(app), host="0.0.0.0", port=cfg.dashboard_port,
                                           log_level="warning", timeout_graceful_shutdown=3))
    # uvicorn shuts down gracefully on SIGINT/SIGTERM, then re-raises the signal with the previous handler.
    # Make that handler raise SystemExit so the `finally` below runs (default SIGTERM would just kill us).
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    try:
        server.run()
    except KeyboardInterrupt:
        pass  # uvicorn re-raises Ctrl+C after its own graceful shutdown
    finally:
        app.stop()


if __name__ == "__main__":
    main()

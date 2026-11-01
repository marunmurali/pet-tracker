"""Pan/tilt follow controller and the absent/resting/active activity state machine.

Both are pure logic driven by `update(t, detection)` so they can be tested with synthetic sequences.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .detector import Detection

ABSENT, RESTING, ACTIVE = "absent", "resting", "active"

log = logging.getLogger(__name__)


def frame_signature(frame) -> "np.ndarray":
    """Tiny blurred grayscale thumbnail, to tell whether the camera actually moved."""
    import cv2

    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (5, 5), 0).astype("float32")


class PanTiltController:
    """Deadband P-controller on the bbox centre that issues short pan/tilt pulses.

    After each pulse the image is moving, so `camera_moving(t)` is True for the pulse plus `settle_s`;
    the controller doesn't re-aim and the activity tracker ignores motion during that time.
    The Tapo's video runs ~1.1 s behind (measured), hence the long default settle time.

    When the dog has been lost for `lost_timeout_s`, it sweeps horizontally: a pan step, a pause so the
    detector gets still frames, repeat (a C200 step at full speed for 1 s is ~30 degrees). A step that doesn't
    change the pan position (from ONVIF) means the pan hit its end stop, so the sweep reverses. The Tapo
    ignores the first move after hitting a stop, so right after reversing a non-move is retried, not
    taken as the other end. Backends without a position compare thumbnails instead. It ends when he's seen again or after `sweep_max_s` (then home preset).
    """

    def __init__(self, ptz, frame_w: int, frame_h: int, *, deadband: float = 0.12, gain: float = 1.2,
                 min_speed: float = 0.25, pulse_s: float = 0.25, settle_s: float = 1.5,
                 lost_timeout_s: float = 8.0, home_preset: str = "", sweep: bool = True,
                 sweep_speed: float = 1.0, sweep_pulse_s: float = 1.0, sweep_dwell_s: float = 2.0,
                 sweep_max_s: float = 120.0, sweep_min_gap_s: float = 300.0, still_diff: float = 3.0):
        self.ptz = ptz
        self.w, self.h = frame_w, frame_h
        self.deadband, self.gain, self.min_speed = deadband, gain, min_speed
        self.pulse_s, self.settle_s = pulse_s, settle_s
        self.lost_timeout_s = lost_timeout_s
        self.home_preset = home_preset
        self.settle_until = 0.0
        self.last_seen: Optional[float] = None  # None until the first update
        self.lost_handled = False  # not seen since start counts as lost too, so the sweep can find him
        self.at_home = True  # zones are only valid while the camera hasn't been moved away from home
        self.sweep_enabled = sweep
        self.sweep_speed, self.sweep_pulse_s, self.sweep_dwell_s = sweep_speed, sweep_pulse_s, sweep_dwell_s
        self.sweep_max_s = sweep_max_s
        self.sweep_min_gap_s = sweep_min_gap_s  # spare the motor: no sweep sooner than this after the last
        self.still_diff = still_diff  # mean abs thumbnail difference below which "the camera didn't move"
        self.sweeping = False
        self._sweep_dir = 1
        self._last_side = 1  # which half of the picture he was last seen in: sweep that way first
        self._sweep_started = float("-inf")
        self._sweep_pending = False  # lost, but too soon after the last sweep: start when the gap is over
        self._next_step = 0.0
        self._before = None  # thumbnail taken just before the last sweep step
        self._stuck = 0
        self._steps = 0
        self._just_reversed = False

    def camera_moving(self, t: float) -> bool:
        return t < self.settle_until

    def _axis(self, err: float) -> float:
        if abs(err) < self.deadband:
            return 0.0
        v = max(self.min_speed, min(1.0, self.gain * abs(err) * 2))
        return math.copysign(v, err)

    def _go_home(self, t: float) -> None:
        if self.home_preset:
            self.ptz.goto_preset(self.home_preset)
            self.at_home = True
            self.settle_until = t + 3.0

    def stop_sweep(self, t: float, go_home: bool = False) -> None:
        if self.sweeping:
            self.sweeping = False
            self.ptz.stop()
            if go_home:
                self._go_home(t)

    def _sweep_step(self, t: float, frame) -> Optional[tuple[float, float]]:
        if t - self._sweep_started > self.sweep_max_s:
            self.stop_sweep(t, go_home=True)
            return None
        if t < self._next_step:
            return None
        moved = getattr(self.ptz, "last_pulse_moved", None)  # result of the previous step, from the camera
        sig = frame_signature(frame) if frame is not None else None
        if moved is None and self._steps and getattr(self.ptz, "position", None) is None:
            # this backend reports no position: did the picture change? (video lags ~1 s; the dwell covers it)
            if sig is not None and self._before is not None:
                moved = float(abs(sig - self._before).mean()) >= self.still_diff
        if moved is False:
            self._stuck += 1
            if self._stuck >= 4:  # doesn't move either way: pan/tilt isn't working
                self.stop_sweep(t)
                return None
            if self._just_reversed:
                self._just_reversed = False  # first move after a stop is ignored by the camera: retry it
            else:
                self._sweep_dir = -self._sweep_dir  # end stop reached: go the other way
                self._just_reversed = True
                log.info("sweep: end stop, reversing")
        elif moved:
            self._stuck, self._just_reversed = 0, False
        self._before = sig
        self._steps += 1
        vx = self._sweep_dir * self.sweep_speed
        self.ptz.pulse(vx, 0.0, self.sweep_pulse_s)
        self.at_home = False
        self.settle_until = t + self.sweep_pulse_s + self.settle_s
        self._next_step = t + self.sweep_pulse_s + self.sweep_dwell_s
        return vx, 0.0

    def update(self, t: float, det: Optional[Detection], frame=None) -> Optional[tuple[float, float]]:
        """Returns the (vx, vy) pulse sent, if any. `frame` is needed for sweeping."""
        if self.last_seen is None:
            self.last_seen = t
        if det is None:
            if not self.lost_handled and t - self.last_seen > self.lost_timeout_s:
                self.lost_handled = True
                self.ptz.stop()
                if self.sweep_enabled:
                    self._sweep_pending = True
                else:
                    self._go_home(t)
            if self._sweep_pending and t - self._sweep_started >= self.sweep_min_gap_s:
                # one sweep per disappearance; first step a moment after Stop (a move sent with Stop can be lost)
                self._sweep_pending = False
                self.sweeping, self._sweep_started, self._next_step = True, t, t + 1.0
                self._before, self._stuck, self._sweep_dir = None, 0, self._last_side
                self._steps, self._just_reversed = 0, False
            if self.sweeping:
                return self._sweep_step(t, frame)
            return None
        self._sweep_pending = False
        if self.sweeping:
            self.sweeping = False  # found him; follow from here
        self.last_seen = t
        self.lost_handled = False
        self._last_side = -1 if det.center[0] < self.w / 2 else 1
        if self.camera_moving(t):
            return None
        cx, cy = det.center
        vx = self._axis(cx / self.w - 0.5)
        vy = -self._axis(cy / self.h - 0.5)  # image y grows downward; +vy tilts up
        if not vx and not vy:
            return None
        self.ptz.pulse(vx, vy, self.pulse_s)
        self.at_home = False
        self.settle_until = t + self.pulse_s + self.settle_s
        return vx, vy


@dataclass
class ActivityParams:
    absent_after_s: float = 10.0  # not seen this long -> absent
    active_on: float = 0.25  # motion (bbox diagonals per second, smoothed) to become active
    active_off: float = 0.10  # ...and to drop back to resting (hysteresis)
    ema_tau_s: float = 2.0
    max_speed: float = 1.0  # cap per-sample speed so one jumpy bbox can't dominate the average
    min_active_s: float = 3.0  # a candidate state must hold this long before switching
    min_resting_s: float = 10.0
    zone_min_s: float = 3.0


class ActivityTracker:
    """absent | resting | active from bbox presence and centroid/size motion.

    `on_change(old, new, t_start)` fires on each committed transition; `t_start` is when the new state
    actually began (start of the candidate period), not when it was confirmed.
    `on_zone(zone_or_None, t)` fires when the dog settles into / leaves a named zone (home position only).
    """

    def __init__(self, params: ActivityParams | None = None,
                 on_change: Callable[[str, str, float], None] | None = None,
                 zones: dict | None = None,
                 on_zone: Callable[[Optional[str], float], None] | None = None):
        self.p = params or ActivityParams()
        self.on_change = on_change or (lambda *a: None)
        self.on_zone = on_zone or (lambda *a: None)
        self.zones = zones or {}
        self.reset()

    def reset(self) -> None:
        self.state = ABSENT
        self.since = 0.0
        self.motion = 0.0
        self.last_seen = -1e9
        self._prev: Optional[tuple[float, Detection]] = None
        self._pending: Optional[str] = None
        self._pending_since = 0.0
        self.zone: Optional[str] = None
        self._zone_pending: Optional[str] = None
        self._zone_pending_since = 0.0

    def update(self, t: float, det: Optional[Detection], frame_w: int, frame_h: int,
               camera_moving: bool = False, at_home: bool = True) -> str:
        if det is not None:
            self.last_seen = t
            self._update_motion(t, det, camera_moving)
        desired = self._desired(t, det)
        self._debounce(t, desired)
        self._update_zone(t, det, frame_w, frame_h, at_home and not camera_moving)
        return self.state

    def _update_motion(self, t: float, det: Detection, camera_moving: bool) -> None:
        if camera_moving:
            self._prev = None  # the whole image shifts; don't count it as the dog moving
            return
        if self._prev is not None:
            pt, pd = self._prev
            dt = t - pt
            if 0 < dt < 2.0:
                diag = math.hypot(det.x2 - det.x1, det.y2 - det.y1) or 1.0
                (cx, cy), (pcx, pcy) = det.center, pd.center
                dw = abs((det.x2 - det.x1) - (pd.x2 - pd.x1))
                dh = abs((det.y2 - det.y1) - (pd.y2 - pd.y1))
                speed = (math.hypot(cx - pcx, cy - pcy) + 0.5 * (dw + dh)) / diag / dt
                speed = min(speed, self.p.max_speed)
                a = 1 - math.exp(-dt / self.p.ema_tau_s)
                self.motion += a * (speed - self.motion)
        self._prev = (t, det)

    def _desired(self, t: float, det: Optional[Detection]) -> str:
        if t - self.last_seen >= self.p.absent_after_s:
            return ABSENT
        if self.state == ABSENT and det is None:
            return ABSENT  # need an actual detection to come back
        if self.motion >= self.p.active_on:
            return ACTIVE
        if self.motion <= self.p.active_off:
            return RESTING
        return self.state if self.state != ABSENT else RESTING

    def _debounce(self, t: float, desired: str) -> None:
        if desired == self.state:
            self._pending = None
            return
        if desired == ABSENT:  # absence already waited absent_after_s
            self._commit(ABSENT, self.last_seen)
            return
        if self._pending != desired:
            self._pending, self._pending_since = desired, t
        min_s = self.p.min_active_s if desired == ACTIVE else self.p.min_resting_s
        if self.state == ABSENT:
            min_s = min(min_s, 1.0)  # arrival is obvious; only filter single-frame false positives
        if t - self._pending_since >= min_s:
            self._commit(desired, self._pending_since)

    def _commit(self, new: str, t_start: float) -> None:
        old = self.state
        t_start = max(t_start, self.since)
        self.state, self.since, self._pending = new, t_start, None
        if new == ABSENT:
            self.motion = 0.0
            self._prev = None
        self.on_change(old, new, t_start)

    def _zone_of(self, det: Detection, w: int, h: int) -> Optional[str]:
        cx, cy = det.center
        x, y = cx / w, cy / h
        for name, (x1, y1, x2, y2) in self.zones.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return name
        return None

    def _update_zone(self, t: float, det: Optional[Detection], w: int, h: int, valid: bool) -> None:
        if not self.zones:
            return
        if not valid:
            current = None
        elif det is None:
            if t - self.last_seen < self.p.absent_after_s:
                return  # brief dropout; keep the zone
            current = None
        else:
            current = self._zone_of(det, w, h)
        if current == self.zone:
            self._zone_pending = None
            return
        if self._zone_pending != current:
            self._zone_pending, self._zone_pending_since = current, t
        if t - self._zone_pending_since >= self.p.zone_min_s or not valid:
            self.zone, self._zone_pending = current, None
            self.on_zone(current, self._zone_pending_since if valid else t)


def iou(a: Detection, b: Detection) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    return inter / (a.area + b.area - inter + 1e-9)


class DogLock:
    """Locks on to the dog at a confident detection and keeps following him where YOLO alone loses him.

    Each update combines a visual single-object tracker (OpenCV ViT) with YOLO run at a low threshold:
      * a confident YOLO box (>= high_conf) always wins, also when it's away from the track (he moved);
      * a weak one (>= weak_conf) overlapping the track confirms it and re-centres the tracker;
      * otherwise the visual tracker carries him, if it is sure (score >= min_score), its box hasn't
        ballooned or shrunk, and either he was confirmed recently or the box is standing still
        (asleep far away, where YOLO never fires).
    Thresholds come from real footage: ViT scores 0.69-0.79 (p25-p50) on the dog, never > 0.59 off it.
    The lock is released after `lost_after_s` without him, and re-taken at the next confident detection.
    """

    def __init__(self, model_path, low_conf: float = 0.10, high_conf: float = 0.45, weak_conf: float = 0.2,
                 lock_conf: float = 0.6,
                 min_score: float = 0.6, max_unconfirmed_s: float = 30.0, lost_after_s: float = 15.0):
        self.model_path = model_path
        self.low_conf, self.high_conf, self.weak_conf = low_conf, high_conf, weak_conf
        self.lock_conf = lock_conf
        self.min_score = min_score
        self.max_unconfirmed_s = max_unconfirmed_s
        self.lost_after_s = lost_after_s
        self._tracker = None
        self.box: Optional[Detection] = None
        self.locked = False
        self.last_seen = 0.0
        self.source = ""
        self._confirmed: Optional[Detection] = None  # last box backed by YOLO or by the user
        self._confirmed_t = 0.0

    def _new_tracker(self):
        import cv2

        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # "Targets are not supported" noise
        if self.model_path and Path(self.model_path).exists():
            p = cv2.TrackerVit_Params()
            p.net = str(self.model_path)
            return cv2.TrackerVit_create(p)
        return cv2.TrackerMIL_create()  # no model file: slower, drift-prone fallback

    def _init(self, frame, box: Detection) -> None:
        if self._tracker is None:
            self._tracker = self._new_tracker()  # loads the model once; init() below can be called again
        x, y = int(box.x1), int(box.y1)
        self._tracker.init(frame, (x, y, max(2, int(box.x2) - x), max(2, int(box.y2) - y)))
        self.box = box

    def _confirm(self, box: Detection, t: float) -> None:
        self._confirmed, self._confirmed_t = box, t

    def lock(self, frame, box: Detection, t: float) -> None:
        self._init(frame, box)
        self._confirm(box, t)
        self.locked, self.last_seen, self.source = True, t, "yolo"

    def unlock(self) -> None:
        self.locked, self.box, self.source, self._confirmed = False, None, "", None

    def _trust_tracker(self, t: float, tracked: Detection) -> bool:
        ref = self._confirmed
        if ref is None or not ref.area:
            return False
        if not 0.5 <= tracked.area / ref.area <= 2.0:
            return False  # ballooned or collapsed: drifting onto background
        if t - self._confirmed_t <= self.max_unconfirmed_s:
            return True
        (cx, cy), (rx, ry) = tracked.center, ref.center
        diag = math.hypot(ref.x2 - ref.x1, ref.y2 - ref.y1)
        return math.hypot(cx - rx, cy - ry) < 0.15 * diag  # long unconfirmed is only OK while he lies still

    def update(self, t: float, frame, dets: list[Detection]) -> Optional[Detection]:
        """dets: YOLO detections at >= low_conf. Returns the dog's box this frame, if known."""
        confident = [d for d in dets if d.conf >= self.high_conf]
        if not self.locked:
            best = max(confident, key=lambda d: d.conf, default=None)
            if best is not None and best.conf >= self.lock_conf:
                self.lock(frame, best, t)
            self.source = "yolo" if best else ""
            return best

        tracked = None
        ok, rect = self._tracker.update(frame)
        score = self._tracker.getTrackingScore() if hasattr(self._tracker, "getTrackingScore") else 1.0
        if ok and score >= self.min_score:
            x, y, w, h = rect
            tracked = Detection(float(x), float(y), float(x + w), float(y + h), float(score))
            if not self._trust_tracker(t, tracked):
                tracked = None

        ref = tracked or self.box
        weak = [d for d in dets if d.conf >= self.weak_conf]
        match = max(weak, key=lambda d: iou(d, ref), default=None) if ref else None
        if match is not None and iou(match, ref) < 0.3:
            match = None
        if confident and (match is None or match.conf < self.high_conf):
            found, self.source = max(confident, key=lambda d: d.conf), "yolo"
        elif match is not None:
            found, self.source = match, "yolo+lock"
        elif tracked is not None:
            found, self.source = tracked, "lock"
        else:
            if t - self.last_seen > self.lost_after_s:
                self.unlock()
            return None
        if found is not tracked:
            self._init(frame, found)  # re-centre the visual tracker on the detector's box
            self._confirm(found, t)
        self.box, self.last_seen = found, t
        return found

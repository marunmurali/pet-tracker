"""Pan/tilt backends (ONVIF or pytapo) behind one non-blocking interface.

Velocities are in -1..1; +x pans right, +y tilts up. Commands go through a worker thread so a slow
camera never stalls the detection loop; a newer command replaces one that hasn't been sent yet.
"""
from __future__ import annotations

import logging
import queue
import threading
import time

log = logging.getLogger(__name__)


class Backend:
    def pulse(self, vx: float, vy: float, duration: float) -> None: ...
    def stop(self) -> None: ...
    def goto_preset(self, preset: str) -> None: ...

    def position(self) -> tuple[float, float] | None:
        """Pan/tilt position, if the camera reports one."""
        return None


class OnvifBackend(Backend):
    def __init__(self, host: str, port: int, user: str, password: str):
        from onvif import ONVIFCamera  # imported lazily: zeep is slow to import

        cam = ONVIFCamera(host, port, user, password)
        self.ptz = cam.create_ptz_service()
        self.token = cam.create_media_service().GetProfiles()[0].token

    def pulse(self, vx, vy, duration):
        self.ptz.ContinuousMove({"ProfileToken": self.token, "Velocity": {"PanTilt": {"x": vx, "y": vy}}})
        time.sleep(duration)
        self.stop()

    def stop(self):
        self.ptz.Stop({"ProfileToken": self.token, "PanTilt": True, "Zoom": False})

    def goto_preset(self, preset):
        self.ptz.GotoPreset({"ProfileToken": self.token, "PresetToken": preset})

    def position(self):
        try:
            p = self.ptz.GetStatus({"ProfileToken": self.token}).Position.PanTilt
            return float(p.x), float(p.y)
        except Exception:
            return None


class PytapoBackend(Backend):
    DEG_PER_UNIT_SECOND = 40  # full speed for 1 s ~= 40 degrees; moveMotor takes relative steps

    def __init__(self, host: str, cloud_password: str):
        from pytapo import Tapo

        self.tapo = Tapo(host, "admin", cloud_password)

    def pulse(self, vx, vy, duration):
        dx = round(vx * duration * self.DEG_PER_UNIT_SECOND)
        dy = round(vy * duration * self.DEG_PER_UNIT_SECOND)
        if dx or dy:
            self.tapo.moveMotor(dx, dy)

    def stop(self):
        pass  # moveMotor is a relative step; nothing keeps moving

    def goto_preset(self, preset):
        self.tapo.setPreset(preset)


class PTZ:
    """Thread-safe front end. `available` is False until the backend connects (retried lazily)."""

    def __init__(self, kind: str, host: str, onvif_port: int, user: str, password: str, cloud_password: str,
                 invert_pan: bool = False, invert_tilt: bool = False):
        self.kind = kind
        self._sign = (-1 if invert_pan else 1, -1 if invert_tilt else 1)
        self._args = (host, onvif_port, user, password, cloud_password)
        self._backend: Backend | None = None
        self._next_connect = 0.0
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # After each pulse: did the position change? None = unknown (pending, or no position reported).
        # False means the axis is at its end stop. Used by the sweep in tracker.py.
        self.last_pulse_moved: bool | None = None
        self.position: tuple[float, float] | None = None
        self.moves_today = 0  # motor use, to keep an eye on wear; reset at local midnight
        self.motor_s_today = 0.0
        self._day = time.strftime("%Y-%m-%d")
        if kind != "none":
            self._thread = threading.Thread(target=self._run, name="ptz", daemon=True)
            self._thread.start()

    @property
    def enabled(self) -> bool:
        return self.kind != "none"

    @property
    def available(self) -> bool:
        return self._backend is not None

    def pulse(self, vx: float, vy: float, duration: float) -> None:
        self.last_pulse_moved = None
        self._submit(("pulse", (vx * self._sign[0], vy * self._sign[1], duration)))

    def stop(self) -> None:
        self._submit(("stop", ()))

    def goto_preset(self, preset: str) -> None:
        self._submit(("preset", (preset,)))

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._submit(("quit", ()))
            self._thread.join(timeout=3)

    def _pulse(self, backend: Backend, vx: float, vy: float, duration: float) -> None:
        day = time.strftime("%Y-%m-%d")
        if day != self._day:
            log.info("pan/tilt yesterday: %d moves, %.0f s of motor time", self.moves_today, self.motor_s_today)
            self._day, self.moves_today, self.motor_s_today = day, 0, 0.0
        self.moves_today += 1
        self.motor_s_today += duration
        before = backend.position()
        backend.pulse(vx, vy, duration)
        after = backend.position()
        if before is not None and after is not None:
            self.position = after
            self.last_pulse_moved = max(abs(after[0] - before[0]), abs(after[1] - before[1])) > 0.005

    def _submit(self, cmd) -> None:
        if not self.enabled:
            return
        try:
            self._q.get_nowait()  # drop a stale pending command
        except queue.Empty:
            pass
        try:
            self._q.put_nowait(cmd)
        except queue.Full:
            pass

    def _connect(self) -> Backend | None:
        if self._backend or time.monotonic() < self._next_connect:
            return self._backend
        host, port, user, password, cloud_password = self._args
        try:
            if self.kind == "onvif":
                self._backend = OnvifBackend(host, port, user, password)
            elif self.kind == "pytapo":
                self._backend = PytapoBackend(host, cloud_password)
            else:
                raise ValueError(f"unknown PTZ_BACKEND {self.kind!r} (use onvif, pytapo or none)")
            log.info("pan/tilt connected via %s", self.kind)
        except Exception as e:
            log.warning("pan/tilt %s unavailable (%s); retrying in 60 s", self.kind, type(e).__name__)
            self._next_connect = time.monotonic() + 60
        return self._backend

    def _run(self) -> None:
        while not self._stop.is_set():
            op, args = self._q.get()
            if op == "quit":
                return
            backend = self._connect()
            if not backend:
                continue
            try:
                if op == "pulse":
                    self._pulse(backend, *args)
                else:
                    {"stop": backend.stop, "preset": backend.goto_preset}[op](*args)
            except Exception as e:
                log.warning("pan/tilt %s failed: %s", op, e)
                self._backend = None  # reconnect on next command

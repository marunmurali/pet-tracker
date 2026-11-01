from app.detector import Detection
from app.tracker import ABSENT, ACTIVE, RESTING, ActivityParams, ActivityTracker, PanTiltController

W, H = 640, 360


class FakePTZ:
    def __init__(self):
        self.calls = []

    def pulse(self, vx, vy, duration):
        self.calls.append(("pulse", vx, vy))

    def stop(self):
        self.calls.append(("stop",))

    def goto_preset(self, preset):
        self.calls.append(("preset", preset))


def box(cx, cy, w=100, h=80):
    return Detection(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 0.9)


# --- pan/tilt controller ---------------------------------------------------------------------------

def test_no_move_inside_deadband():
    ptz = FakePTZ()
    c = PanTiltController(ptz, W, H)
    assert c.update(0, box(330, 190)) is None
    assert ptz.calls == []


def test_moves_toward_dog_then_settles():
    ptz = FakePTZ()
    c = PanTiltController(ptz, W, H, pulse_s=0.25, settle_s=0.8)
    vx, vy = c.update(0, box(600, 50))  # right edge, top
    assert vx > 0 and vy > 0  # pan right, tilt up
    assert c.camera_moving(0.5)
    assert c.update(0.5, box(600, 50)) is None  # ignored while settling
    assert not c.camera_moving(1.1)
    assert c.update(1.1, box(600, 50)) is not None
    assert not c.at_home


def test_lost_dog_stops_and_returns_home():
    ptz = FakePTZ()
    c = PanTiltController(ptz, W, H, lost_timeout_s=5, home_preset="1", sweep=False)
    c.update(0, box(320, 180))
    c.update(3, None)
    assert ptz.calls == []
    c.update(6, None)
    assert ptz.calls == [("stop",), ("preset", "1")]
    c.update(7, None)
    assert len(ptz.calls) == 2  # only once
    assert c.at_home


class SweepCam:
    """Fake pan axis with end stops at +-3 steps; the picture only changes when the pan really moves."""

    def __init__(self):
        self.pos, self.calls = 0, []

    def pulse(self, vx, vy, duration):
        self.calls.append(vx)
        self.pos = max(-3, min(3, self.pos + (1 if vx > 0 else -1)))

    def stop(self):
        pass

    def goto_preset(self, preset):
        self.calls.append(("preset", preset))

    def frame(self):
        import numpy as np
        f = np.zeros((H, W, 3), np.uint8)
        f[:, (self.pos + 4) * 60:(self.pos + 4) * 60 + 40] = 255  # a bright post that shifts as we pan
        return f


def run_sweep(c, cam, t0, t1, step=0.5):
    t = t0
    while t < t1:
        c.update(t, None, cam.frame())
        t += step
    return t


def test_sweeps_when_lost_and_bounces_off_end_stops():
    cam = SweepCam()
    c = PanTiltController(cam, W, H, lost_timeout_s=5, sweep_max_s=1000)
    c.update(0, box(320, 180))
    run_sweep(c, cam, 1, 4)
    assert cam.calls == [] and not c.sweeping  # not lost yet
    run_sweep(c, cam, 6, 60)
    assert c.sweeping
    dirs = [1 if v > 0 else -1 for v in cam.calls]
    flips = sum(a != b for a, b in zip(dirs, dirs[1:]))
    assert flips >= 2  # reached one end, came back, reached the other
    c.update(61, box(320, 180), cam.frame())
    assert not c.sweeping  # found him


def test_sweep_starts_toward_where_he_was_last_seen():
    cam = SweepCam()
    c = PanTiltController(cam, W, H, lost_timeout_s=5, sweep_max_s=1000)
    c.update(0, box(40, 180))  # left edge of the picture (this also pans left once, to follow him)
    n = len(cam.calls)
    run_sweep(c, cam, 6, 9)
    assert len(cam.calls) > n and cam.calls[n] < 0  # the sweep's first step goes left too


def test_sweep_gives_up_and_goes_home():
    cam = SweepCam()
    c = PanTiltController(cam, W, H, lost_timeout_s=5, sweep_max_s=20, home_preset="1")
    c.update(0, box(320, 180))
    run_sweep(c, cam, 6, 40)
    assert not c.sweeping and cam.calls[-1] == ("preset", "1")


def test_sweeps_after_start_if_never_seen():
    cam = SweepCam()
    c = PanTiltController(cam, W, H, lost_timeout_s=5, sweep_max_s=1000)
    run_sweep(c, cam, 100, 104)
    assert not c.sweeping  # give him lost_timeout_s to show up first
    run_sweep(c, cam, 104, 110)
    assert c.sweeping and cam.calls


class TapoLikeCam(SweepCam):
    """Reports its position (like ONVIF GetStatus) and ignores the first move after hitting an end stop."""

    def __init__(self):
        super().__init__()
        self.position, self.last_pulse_moved, self._at_stop = (0.0, 0.0), None, False
        self.visited = set()

    def pulse(self, vx, vy, duration):
        self.calls.append(vx)
        if self._at_stop:
            self._at_stop, self.last_pulse_moved = False, False  # swallowed
            return
        new = max(-3, min(3, self.pos + (1 if vx > 0 else -1)))
        self.last_pulse_moved = new != self.pos
        self._at_stop = not self.last_pulse_moved
        self.pos = new
        self.position = (float(new), 0.0)
        self.visited.add(new)


def test_sweep_uses_position_and_survives_the_ignored_move_after_a_stop():
    cam = TapoLikeCam()
    c = PanTiltController(cam, W, H, lost_timeout_s=5, sweep_max_s=1000)
    c.update(0, box(320, 180))
    run_sweep(c, cam, 6, 200)
    assert c.sweeping  # never mistook the swallowed move for "stuck"
    assert cam.visited == set(range(-3, 4))  # covered the whole travel
    dirs = [1 if v > 0 else -1 for v in cam.calls]
    assert sum(a != b for a, b in zip(dirs, dirs[1:])) >= 4  # bounced between the ends several times


def test_no_new_sweep_within_min_gap():
    cam = SweepCam()
    c = PanTiltController(cam, W, H, lost_timeout_s=5, sweep_max_s=20, sweep_min_gap_s=300)
    c.update(0, box(320, 180))
    run_sweep(c, cam, 6, 40)  # sweeps, gives up
    c.update(50, box(320, 180), cam.frame())  # a brief sighting (maybe a false one)...
    n = len(cam.calls)
    run_sweep(c, cam, 51, 120)  # ...then lost again: too soon for another sweep
    assert not c.sweeping and len(cam.calls) == n
    run_sweep(c, cam, 320, 330)
    assert c.sweeping


def test_sweep_stops_if_camera_never_moves():
    class Stuck(SweepCam):
        def pulse(self, vx, vy, duration):
            self.calls.append(vx)  # commands go out but nothing moves
    cam = Stuck()
    c = PanTiltController(cam, W, H, lost_timeout_s=5, sweep_max_s=1000)
    c.update(0, box(320, 180))
    run_sweep(c, cam, 6, 60)
    assert not c.sweeping and len(cam.calls) <= 5


# --- activity state machine ------------------------------------------------------------------------

def run(tracker, seq, fps=5, t0=0.0, **kw):
    t = t0
    for det in seq:
        tracker.update(t, det, W, H, **kw)
        t += 1 / fps
    return t


def test_arrival_rest_activity_and_absence():
    changes = []
    tr = ActivityTracker(ActivityParams(), on_change=lambda o, n, t: changes.append((o, n, round(t, 1))))
    t = run(tr, [box(320, 180)] * 5 * 20)  # still for 20 s
    assert tr.state == RESTING
    t = run(tr, [box(200 + (i % 2) * 120, 180) for i in range(5 * 10)], t0=t)  # running back and forth
    assert tr.state == ACTIVE
    t = run(tr, [None] * 5 * 12, t0=t)  # gone for 12 s
    assert tr.state == ABSENT
    assert [c[1] for c in changes] == [RESTING, ACTIVE, ABSENT]
    # absence is backdated to when the dog was last seen (t = 30 s)
    assert changes[-1][2] == 29.8


def test_single_false_positive_does_not_count_as_arrival():
    tr = ActivityTracker()
    run(tr, [None] * 10 + [box(320, 180)] + [None] * 50)
    assert tr.state == ABSENT


def test_short_twitch_does_not_flip_to_active():
    tr = ActivityTracker()
    t = run(tr, [box(320, 180)] * 100)
    assert tr.state == RESTING
    t = run(tr, [box(320 + (i % 2) * 60, 180) for i in range(4)], t0=t)  # <1 s of movement
    t = run(tr, [box(320, 180)] * 25, t0=t)
    assert tr.state == RESTING


def test_camera_motion_is_ignored():
    tr = ActivityTracker()
    t = run(tr, [box(320, 180)] * 100)
    # the dog appears to jump around only because the camera is panning
    t = run(tr, [box(100 + i * 40, 180) for i in range(10)], t0=t, camera_moving=True)
    assert tr.state == RESTING and tr.motion < 0.1


def test_zones_only_at_home():
    zones = {"bed": (0.0, 0.5, 0.5, 1.0)}
    seen = []
    tr = ActivityTracker(zones=zones, on_zone=lambda z, t: seen.append(z))
    t = run(tr, [box(100, 300)] * 25)  # 5 s in the bed area
    assert seen == ["bed"] and tr.zone == "bed"
    run(tr, [box(100, 300)] * 5, t0=t, at_home=False)
    assert seen == ["bed", None]

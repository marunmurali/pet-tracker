import numpy as np

from app.detector import Detection
from app.tracker import DogLock
from app.training import TrainingCollector

FRAME = np.zeros((360, 640, 3), np.uint8)


def box(x, y, w=80, h=60, conf=0.9):
    return Detection(x, y, x + w, y + h, conf)


class FakeTracker:
    """Stands in for cv2.TrackerVit: reports whatever box the test sets."""

    def __init__(self):
        self.rect, self.score, self.inits = None, 0.9, 0

    def init(self, frame, rect):
        self.rect, self.inits = rect, self.inits + 1

    def update(self, frame):
        return self.rect is not None, self.rect or (0, 0, 0, 0)

    def getTrackingScore(self):
        return self.score


def make_lock(**kw):
    lock = DogLock(None, **kw)
    fake = FakeTracker()
    lock._new_tracker = lambda: fake
    return lock, fake


def test_unlocked_uses_confident_yolo_only():
    lock, _ = make_lock()
    assert lock.update(0, FRAME, [box(10, 10, conf=0.2)]) is None
    assert lock.update(0, FRAME, [box(10, 10, conf=0.5)]).conf == 0.5
    assert not lock.locked  # 0.5 is enough to report him, not to lock on


def test_locks_on_automatically_at_a_sure_detection():
    lock, fake = make_lock()
    lock.update(0, FRAME, [box(10, 10, conf=0.8)])
    assert lock.locked and fake.inits == 1
    fake.rect = (12, 10, 80, 60)
    assert lock.update(1, FRAME, []).x1 == 12 and lock.source == "lock"  # followed without YOLO


def test_weak_detection_on_the_track_is_accepted_and_recentres():
    lock, fake = make_lock()
    lock.lock(FRAME, box(100, 100), 0)
    weak = box(104, 102, conf=0.25)  # below the normal threshold, but where the dog is
    assert lock.update(1, FRAME, [weak]) is weak
    assert lock.source == "yolo+lock" and fake.inits == 2


def test_noise_level_detection_does_not_confirm():
    lock, fake = make_lock()
    lock.lock(FRAME, box(100, 100), 0)
    fake.score = 0.1
    assert lock.update(1, FRAME, [box(104, 102, conf=0.15)]) is None  # e.g. a table leg at 0.15


def test_confident_detection_beats_weak_match_on_track():
    lock, fake = make_lock()
    lock.lock(FRAME, box(100, 100), 0)
    weak, strong = box(102, 100, conf=0.21), box(495, 205, conf=0.83)
    assert lock.update(1, FRAME, [weak, strong]) is strong


def test_ballooning_tracker_box_is_not_trusted():
    lock, fake = make_lock()
    lock.lock(FRAME, box(100, 100), 0)
    fake.rect = (20, 20, 400, 300)  # 25x the area: drifted onto the background
    assert lock.update(1, FRAME, []) is None


def test_long_unconfirmed_tracking_only_while_still():
    lock, fake = make_lock(max_unconfirmed_s=30, lost_after_s=1000)
    lock.lock(FRAME, box(100, 100), 0)
    fake.rect = (102, 101, 80, 60)  # asleep: barely moves
    assert lock.update(100, FRAME, []) is not None
    fake.rect = (300, 200, 80, 60)  # "moving" with no detector support for 100 s: don't believe it
    assert lock.update(101, FRAME, []) is None


def test_tracker_carries_when_yolo_sees_nothing():
    lock, fake = make_lock()
    lock.lock(FRAME, box(100, 100), 0)
    fake.rect = (120, 100, 80, 60)
    b = lock.update(1, FRAME, [])
    assert lock.source == "lock" and b.x1 == 120


def test_confident_detection_elsewhere_wins():
    lock, fake = make_lock()
    lock.lock(FRAME, box(100, 100), 0)
    far = box(500, 250, conf=0.85)
    assert lock.update(1, FRAME, [far]) is far and lock.source == "yolo"


def test_released_after_losing_him():
    lock, fake = make_lock(lost_after_s=5)
    lock.lock(FRAME, box(100, 100), 0)
    fake.score = 0.1  # tracker unsure
    assert lock.update(3, FRAME, []) is None and lock.locked
    assert lock.update(6, FRAME, []) is None and not lock.locked


def test_training_collector_writes_yolo_labels(tmp_path):
    c = TrainingCollector(tmp_path, interval_s=10)
    frame = np.zeros((360, 640, 3), np.uint8)
    assert c.maybe_save(0, frame, box(320, 180, 64, 36), "yolo") is not None
    assert c.maybe_save(5, frame, box(0, 0), "yolo") is None  # too soon
    assert c.maybe_save(15, frame, box(321, 180, 64, 36), "yolo") is None  # same pose, skip
    assert c.maybe_save(30, frame, box(10, 10), "yolo") is not None  # moved
    labels = sorted((tmp_path / "labels").glob("*.txt"))
    assert len(labels) == 2 and c.count() == 2
    cls, cx, cy, w, h = labels[0].read_text().split()
    assert cls == "0" and float(cx) == 0.55 and float(cy) == 0.55 and float(w) == 0.1


def test_uncertain_pictures_go_to_review(tmp_path):
    c = TrainingCollector(tmp_path)
    frame = np.zeros((360, 640, 3), np.uint8)
    c.save(0, frame, box(10, 10, conf=0.7), "lock")  # visual tracker only
    c.save(1, frame, box(10, 10, conf=0.22), "yolo+lock")  # weak detection
    c.save(2, frame, box(10, 10, conf=0.35), "yolo+lock")
    assert c.count() == 1
    assert len(list((tmp_path / "review" / "images").glob("*.jpg"))) == 2

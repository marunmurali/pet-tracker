import datetime as dt
from zoneinfo import ZoneInfo

from app.store import Store
from app.summary import day_bounds, day_stats, fmt_duration, format_summary

TZ = ZoneInfo("Asia/Dubai")


def test_settings_roundtrip(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    assert s.get("enabled", True) is True
    s.set("enabled", False)
    assert Store(tmp_path / "db.sqlite").get("enabled") is False  # persisted


def test_crash_leaves_no_open_interval(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    s.start_interval("resting", 100.0)
    s2 = Store(tmp_path / "db.sqlite")  # simulates a restart after a crash
    assert s2.intervals(0, 10_000) == []


def test_day_stats(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    day = dt.date(2026, 3, 4)
    start, end = day_bounds(day, TZ)
    s.start_interval("absent", start - 600)  # began the day before; clipped at midnight
    s.start_interval("resting", start + 3600)
    s.start_interval("active", start + 3 * 3600)
    s.start_interval("resting", start + 4 * 3600)
    s.close_interval(start + 4 * 3600 + 1800)
    s.add_event("bark", start + 3700)
    s.add_event("bark", start + 3800)
    s.add_event("bark", end + 10)  # next day
    st = day_stats(s, day, TZ)
    assert st["per_state"] == {"absent": 3600, "resting": 2 * 3600 + 1800, "active": 3600}
    assert st["barks"] == 2
    assert st["longest_rest_s"] == 2 * 3600
    text = format_summary(st, "Rex", TZ)
    assert "Resting: 2h 30m" in text and "Barks: 2" in text and "from 01:00" in text


def test_fmt_duration():
    assert fmt_duration(59 * 60) == "59m"
    assert fmt_duration(3600 + 5 * 60) == "1h 05m"

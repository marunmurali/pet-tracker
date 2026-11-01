"""Daily statistics: time per state, barks, longest rest, and snapshots picked for the summary."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

from .store import Store


def day_bounds(day: dt.date, tz: ZoneInfo) -> tuple[float, float]:
    start = dt.datetime.combine(day, dt.time(), tz)
    return start.timestamp(), (start + dt.timedelta(days=1)).timestamp()


def fmt_duration(seconds: float) -> str:
    m = int(round(seconds / 60))
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def day_stats(store: Store, day: dt.date, tz: ZoneInfo) -> dict:
    start, end = day_bounds(day, tz)
    intervals = store.intervals(start, end)
    per_state: dict[str, float] = {"absent": 0.0, "resting": 0.0, "active": 0.0}
    longest_rest = (0.0, None)
    for iv in intervals:
        d = iv["end"] - iv["start"]
        per_state[iv["state"]] = per_state.get(iv["state"], 0.0) + d
        if iv["state"] == "resting" and d > longest_rest[0]:
            longest_rest = (d, iv["start"])
    return {
        "date": day.isoformat(),
        "per_state": per_state,
        "watched_s": sum(per_state.values()),
        "barks": store.count_events(start, end, "bark"),
        "longest_rest_s": longest_rest[0],
        "longest_rest_start": longest_rest[1],
        "intervals": intervals,
    }


def format_summary(stats: dict, dog_name: str, tz: ZoneInfo) -> str:
    ps = stats["per_state"]
    lines = [f"🐶 {dog_name} — {stats['date']}"]
    if stats["watched_s"] < 60:
        lines.append("Not watched today.")
    else:
        lines += [
            f"Watched: {fmt_duration(stats['watched_s'])}",
            f"😴 Resting: {fmt_duration(ps['resting'])}",
            f"🏃 Active: {fmt_duration(ps['active'])}",
            f"👻 Out of view: {fmt_duration(ps['absent'])}",
        ]
        if stats["longest_rest_start"]:
            at = dt.datetime.fromtimestamp(stats["longest_rest_start"], tz).strftime("%H:%M")
            lines.append(f"Longest rest: {fmt_duration(stats['longest_rest_s'])} (from {at})")
    lines.append(f"🔊 Barks: {stats['barks']}")
    return "\n".join(lines)


def pick_snapshots(snap_dir: Path, day: dt.date, n: int = 4) -> list[Path]:
    """Evenly spaced snapshots from data/snapshots/YYYY-MM-DD/."""
    files = sorted((snap_dir / day.isoformat()).glob("*.jpg"))
    if len(files) <= n:
        return files
    step = (len(files) - 1) / (n - 1)
    return [files[round(i * step)] for i in range(n)]

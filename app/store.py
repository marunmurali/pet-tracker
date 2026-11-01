"""SQLite persistence: settings (shared on/off state), activity intervals, and events (barks, zones, ...)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS intervals (
    id INTEGER PRIMARY KEY,
    state TEXT NOT NULL,
    start REAL NOT NULL,
    end REAL               -- NULL while open
);
CREATE INDEX IF NOT EXISTS intervals_start ON intervals(start);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    type TEXT NOT NULL,    -- bark | zone | power | ...
    data TEXT NOT NULL DEFAULT '{}',
    clip TEXT              -- file name in data/clips/
);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)
            # a crash leaves an interval open; close it at its start so it doesn't span the downtime
            self._db.execute("UPDATE intervals SET end = start WHERE end IS NULL")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _q(self, sql: str, args=()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    # --- settings ----------------------------------------------------------------------------------
    def get(self, key: str, default=None):
        rows = self._q("SELECT value FROM settings WHERE key = ?", (key,))
        return json.loads(rows[0]["value"]) if rows else default

    def set(self, key: str, value) -> None:
        self._q("INSERT INTO settings(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, json.dumps(value)))

    # --- intervals ---------------------------------------------------------------------------------
    def start_interval(self, state: str, ts: float) -> None:
        """Close the open interval at `ts` and open a new one."""
        with self._lock:
            self._db.execute("DELETE FROM intervals WHERE end IS NULL AND start >= ?", (ts,))  # zero-length
            self._db.execute("UPDATE intervals SET end = ? WHERE end IS NULL", (ts,))
            self._db.execute("INSERT INTO intervals(state, start) VALUES (?, ?)", (state, ts))

    def close_interval(self, ts: float) -> None:
        self._q("UPDATE intervals SET end = MAX(start, ?) WHERE end IS NULL", (ts,))

    def intervals(self, start: float, end: float) -> list[dict]:
        """Intervals overlapping [start, end), clipped to it. Open intervals run until now."""
        now = time.time()
        rows = self._q("SELECT state, start, end FROM intervals "
                       "WHERE start < ? AND COALESCE(end, ?) > ? ORDER BY start", (end, now, start))
        out = []
        for r in rows:
            s, e = max(r["start"], start), min(r["end"] if r["end"] is not None else now, end)
            if e > s:
                out.append({"state": r["state"], "start": s, "end": e})
        return out

    # --- events ------------------------------------------------------------------------------------
    def add_event(self, type_: str, ts: float | None = None, data: dict | None = None,
                  clip: str | None = None) -> int:
        with self._lock:
            cur = self._db.execute("INSERT INTO events(ts, type, data, clip) VALUES (?, ?, ?, ?)",
                                   (ts or time.time(), type_, json.dumps(data or {}), clip))
            return int(cur.lastrowid)

    def set_event_clip(self, event_id: int, clip: str) -> None:
        self._q("UPDATE events SET clip = ? WHERE id = ?", (clip, event_id))

    def events(self, start: float, end: float, type_: str | None = None, limit: int = 500) -> list[dict]:
        sql = "SELECT id, ts, type, data, clip FROM events WHERE ts >= ? AND ts < ?"
        args: list = [start, end]
        if type_:
            sql += " AND type = ?"
            args.append(type_)
        sql += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        return [{"id": r["id"], "ts": r["ts"], "type": r["type"], "data": json.loads(r["data"]),
                 "clip": r["clip"]} for r in self._q(sql, args)]

    def count_events(self, start: float, end: float, type_: str) -> int:
        return self._q("SELECT COUNT(*) AS n FROM events WHERE ts >= ? AND ts < ? AND type = ?",
                       (start, end, type_))[0]["n"]

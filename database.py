"""
AbojutoAI — SQLite event store (async via aiosqlite shimmed with threading).
Stores detection events with timestamp, count, confidence, and snapshot path.
"""

import sqlite3
import threading
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("abojuto.db")
DB_PATH = Path("db/events.db")
DB_PATH.parent.mkdir(exist_ok=True)

_local = threading.local()


def _get_conn():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _init_schema(_local.conn)
    return _local.conn


def _init_schema(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            count       INTEGER NOT NULL,
            avg_conf    REAL NOT NULL,
            snapshot    TEXT NOT NULL
        )
    """)
    conn.commit()


def log_event(snapshot_path: str, count: int, confidences: list) -> dict:
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    ts = datetime.now().isoformat(timespec="seconds")
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO events (timestamp, count, avg_conf, snapshot) VALUES (?,?,?,?)",
        (ts, count, avg_conf, snapshot_path),
    )
    conn.commit()
    event_id = cur.lastrowid
    logger.info(f"Event #{event_id} logged — {count} person(s)")
    return {
        "id": event_id,
        "timestamp": ts,
        "count": count,
        "avg_conf": round(avg_conf, 3),
        "snapshot": snapshot_path,
    }


def get_recent_events(limit: int = 50) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_stats() -> dict:
    conn = _get_conn()
    row = conn.execute("""
        SELECT
            COUNT(*)           AS total_events,
            COALESCE(SUM(count), 0)  AS total_detections,
            COALESCE(AVG(avg_conf), 0) AS avg_confidence,
            MAX(timestamp)     AS last_seen
        FROM events
    """).fetchone()
    return dict(row) if row else {}

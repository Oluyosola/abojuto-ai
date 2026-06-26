"""
Vigirix Server — Database Layer (SQLite, async via aiosqlite)
Tables: events, devices, happenings
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite
from dotenv import load_dotenv

load_dotenv()
logger   = logging.getLogger("vigirix.db")
DB_PATH  = Path(os.getenv("DB_PATH", "db/vigirix.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    device_id     TEXT    NOT NULL DEFAULT 'unknown',
    location      TEXT    NOT NULL DEFAULT '',
    detections    TEXT    NOT NULL,   -- JSON array
    threat_level  TEXT    NOT NULL DEFAULT 'LOW',
    snapshot_path TEXT,
    acknowledged  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS devices (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id   TEXT UNIQUE NOT NULL,
    name        TEXT,
    location    TEXT,
    last_seen   TEXT,
    status      TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS happenings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT NOT NULL,
    title      TEXT NOT NULL,
    body       TEXT NOT NULL,
    image_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_ts       ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_threat   ON events(threat_level);
CREATE INDEX IF NOT EXISTS idx_events_device   ON events(device_id);
"""


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(SCHEMA)
        await db.commit()
    logger.info(f"Database ready: {DB_PATH}")


# ── Events ────────────────────────────────────────────────────────────────────

async def insert_event(
    device_id:    str,
    location:     str,
    detections:   list,
    threat_level: str,
    snapshot_path: Optional[str] = None,
) -> int:
    ts = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """INSERT INTO events
               (timestamp, device_id, location, detections, threat_level, snapshot_path)
               VALUES (?,?,?,?,?,?)""",
            (ts, device_id, location, json.dumps(detections), threat_level, snapshot_path),
        )
        await db.commit()
        event_id = cur.lastrowid

    await upsert_device(device_id, location)
    return event_id


async def get_events(limit: int = 50, offset: int = 0,
                     threat: Optional[str] = None, device: Optional[str] = None) -> dict:
    clauses, params = [], []
    if threat:
        clauses.append("threat_level = ?"); params.append(threat)
    if device:
        clauses.append("device_id = ?");    params.append(device)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        count_row = await (await db.execute(
            f"SELECT COUNT(*) FROM events {where}", params
        )).fetchone()
        total = count_row[0]
        rows = await db.execute_fetchall(
            f"SELECT * FROM events {where} ORDER BY id DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        )
    return {
        "items":  [_row_to_event(r) for r in rows],
        "total":  total,
        "limit":  limit,
        "offset": offset,
    }


async def cleanup_old_events(days: int = 30):
    cutoff = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    from datetime import timedelta
    cutoff -= timedelta(days=days)
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "DELETE FROM events WHERE timestamp < ?", (cutoff.isoformat(),)
        )
        await db.commit()
        if cur.rowcount:
            logger.info(f"Cleaned up {cur.rowcount} events older than {days} days")


async def get_event(event_id: int) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            "SELECT * FROM events WHERE id = ?", (event_id,)
        )
    return _row_to_event(rows[0]) if rows else None


async def acknowledge_event(event_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE events SET acknowledged=1 WHERE id=?", (event_id,))
        await db.commit()


async def delete_event(event_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM events WHERE id=?", (event_id,))
        await db.commit()


async def delete_all_events():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM events")
        await db.commit()


async def get_stats() -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        total  = (await (await db.execute("SELECT COUNT(*) FROM events")).fetchone())[0]
        today  = datetime.utcnow().date().isoformat()
        today_c= (await (await db.execute(
            "SELECT COUNT(*) FROM events WHERE timestamp LIKE ?", (f"{today}%",)
        )).fetchone())[0]
        crit   = (await (await db.execute(
            "SELECT COUNT(*) FROM events WHERE threat_level='CRITICAL' AND acknowledged=0"
        )).fetchone())[0]
        high   = (await (await db.execute(
            "SELECT COUNT(*) FROM events WHERE threat_level='HIGH' AND acknowledged=0"
        )).fetchone())[0]
        dev_c  = (await (await db.execute("SELECT COUNT(*) FROM devices")).fetchone())[0]

    return {
        "total_events":   total,
        "today_events":   today_c,
        "active_alerts":  crit + high,
        "devices_online": dev_c,
    }


def _row_to_event(row) -> dict:
    d = dict(row)
    d["detections"] = json.loads(d.get("detections", "[]"))
    return d


# ── Devices ───────────────────────────────────────────────────────────────────

async def upsert_device(device_id: str, location: str = ""):
    ts = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO devices (device_id, location, last_seen, status)
               VALUES (?,?,?,'active')
               ON CONFLICT(device_id) DO UPDATE SET last_seen=excluded.last_seen, status='active'
            """,
            (device_id, location, ts),
        )
        await db.commit()


async def get_devices() -> list:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall("SELECT * FROM devices ORDER BY last_seen DESC")
    return [dict(r) for r in rows]


# ── Happenings ────────────────────────────────────────────────────────────────

async def insert_happening(title: str, body: str, image_path: Optional[str] = None) -> int:
    ts = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "INSERT INTO happenings (timestamp, title, body, image_path) VALUES (?,?,?,?)",
            (ts, title, body, image_path),
        )
        await db.commit()
        return cur.lastrowid


async def get_happenings(limit: int = 10) -> list:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            "SELECT * FROM happenings ORDER BY id DESC LIMIT ?", (limit,)
        )
    return [dict(r) for r in rows]


async def get_happening(happening_id: int) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            "SELECT * FROM happenings WHERE id = ?", (happening_id,)
        )
    return dict(rows[0]) if rows else None


async def delete_happening(happening_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM happenings WHERE id=?", (happening_id,))
        await db.commit()

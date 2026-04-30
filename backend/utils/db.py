"""
Vikas.ai — SQLite Database Utility
Handles call record persistence and OTP storage.
"""

import sqlite3
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("vikas.db")

DB_PATH = "backend/data/vikas.db"


def _conn():
    """Get a new SQLite connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't already exist."""
    conn = _conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS calls (
            id          TEXT PRIMARY KEY,
            phone       TEXT NOT NULL,
            recording   TEXT,
            transcript  TEXT,
            summary     TEXT,
            duration    INTEGER DEFAULT 0,
            created_at  TEXT DEFAULT (datetime('now'))
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS otps (
            phone       TEXT PRIMARY KEY,
            code        TEXT NOT NULL,
            expires_at  TEXT NOT NULL,
            verified    INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()
    logger.info("SQLite database initialized at %s", DB_PATH)


# ── Call Records ────────────────────────────────────────────

def save_call(call_id: str, phone: str, recording: str,
              transcript: str, summary: str, duration: int):
    """Insert or replace a call record."""
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO calls (id, phone, recording, transcript, summary, duration) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (call_id, phone, recording, transcript, summary, duration),
    )
    conn.commit()
    conn.close()
    logger.info("Saved call %s for %s", call_id, phone)


def get_calls_for_phone(phone: str) -> list[dict]:
    """Return all calls for a given phone number, newest first."""
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM calls WHERE phone = ? ORDER BY created_at DESC", (phone,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── OTP Management ─────────────────────────────────────────

def store_otp(phone: str, code: str, ttl_minutes: int = 10):
    """Store an OTP, replacing any previous one for that phone."""
    expires = (datetime.utcnow() + timedelta(minutes=ttl_minutes)).isoformat()
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO otps (phone, code, expires_at, verified) VALUES (?, ?, ?, 0)",
        (phone, code, expires),
    )
    conn.commit()
    conn.close()


def check_otp(phone: str, code: str) -> bool:
    """Verify an OTP. Returns True if valid and not expired."""
    conn = _conn()
    row = conn.execute(
        "SELECT code, expires_at FROM otps WHERE phone = ? AND verified = 0",
        (phone,),
    ).fetchone()

    if not row:
        conn.close()
        return False

    if row["code"] == code and datetime.fromisoformat(row["expires_at"]) > datetime.utcnow():
        conn.execute("UPDATE otps SET verified = 1 WHERE phone = ?", (phone,))
        conn.commit()
        conn.close()
        return True

    conn.close()
    return False

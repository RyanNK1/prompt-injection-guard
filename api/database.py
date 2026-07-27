"""
Minimal SQLite storage for the suggestions box (Step 6 addition).

SQLite is used instead of a full database server because it needs zero
setup (it's part of Python's standard library) and a single file on
disk is more than enough for a low-traffic portfolio demo -- there's
no separate database process to install, configure, or keep running.
"""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "suggestions.db"


def init_db():
    """Create the suggestions table if it doesn't already exist. Safe to
    call every time the app starts -- CREATE TABLE IF NOT EXISTS is a
    no-op if the table is already there."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def insert_suggestion(text: str) -> int:
    """Store one suggestion, return its new row id."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "INSERT INTO suggestions (text, created_at) VALUES (?, ?)",
        (text, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def get_all_suggestions() -> list[dict]:
    """Return every stored suggestion, most recent first."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name, not just index
    rows = conn.execute(
        "SELECT id, text, created_at FROM suggestions ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]

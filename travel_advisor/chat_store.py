"""SQLite-backed persistence for chat threads and messages.

Schema (initialised on first use):

    chat_threads (thread_id PK, title, created_at, updated_at)
    chat_messages (id PK, thread_id FK, role, content, trace, created_at)

`trace` is the JSON-encoded list of tool calls produced by the agent
for the assistant turn — stored as TEXT, decoded on read.

Tied to the rest of the app via `settings.chats_db_path`. Threads with
zero messages are filtered out of `list_threads(...)` so a freshly
created thread only appears in the sidebar after the first turn.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_threads (
    thread_id  TEXT PRIMARY KEY,
    title      TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id  TEXT NOT NULL REFERENCES chat_threads(thread_id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    trace      TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_thread ON chat_messages(thread_id, id);
"""

TITLE_MAX_CHARS = 60


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _truncate_title(text: str) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= TITLE_MAX_CHARS:
        return text
    return text[:TITLE_MAX_CHARS].rstrip() + "…"


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_store(db_path: Path) -> None:
    """Create schema if absent. Idempotent."""
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def save_turn(
    db_path: Path,
    *,
    thread_id: str,
    role: str,
    content: str,
    trace: list[dict] | None = None,
) -> None:
    """Insert one message; create the thread row on first message.

    For the first user message in a thread, the thread's title is
    auto-derived from the truncated message content.
    """
    if role not in ("user", "assistant"):
        raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")
    now = _now_iso()
    trace_json = json.dumps(trace, ensure_ascii=False) if trace else None
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT title FROM chat_threads WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        if row is None:
            title = _truncate_title(content) if role == "user" else None
            conn.execute(
                "INSERT INTO chat_threads (thread_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (thread_id, title, now, now),
            )
        elif row["title"] is None and role == "user":
            conn.execute(
                "UPDATE chat_threads SET title = ?, updated_at = ? WHERE thread_id = ?",
                (_truncate_title(content), now, thread_id),
            )
        else:
            conn.execute(
                "UPDATE chat_threads SET updated_at = ? WHERE thread_id = ?",
                (now, thread_id),
            )
        conn.execute(
            "INSERT INTO chat_messages (thread_id, role, content, trace, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (thread_id, role, content, trace_json, now),
        )
        conn.commit()


def load_thread(db_path: Path, thread_id: str) -> list[dict[str, Any]]:
    """Return all messages for a thread, oldest first."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT role, content, trace FROM chat_messages "
            "WHERE thread_id = ? ORDER BY id ASC",
            (thread_id,),
        ).fetchall()
    return [
        {
            "role": r["role"],
            "content": r["content"],
            "trace": json.loads(r["trace"]) if r["trace"] else [],
        }
        for r in rows
    ]


def list_threads(db_path: Path) -> list[dict[str, Any]]:
    """List threads that have at least one message, most-recent first."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT t.thread_id, t.title, t.updated_at, "
            "       COUNT(m.id) AS n_messages "
            "FROM chat_threads t "
            "JOIN chat_messages m ON m.thread_id = t.thread_id "
            "GROUP BY t.thread_id "
            "ORDER BY t.updated_at DESC"
        ).fetchall()
    return [
        {
            "thread_id": r["thread_id"],
            "title": r["title"] or "(untitled)",
            "updated_at": r["updated_at"],
            "n_messages": r["n_messages"],
        }
        for r in rows
    ]


def delete_thread(db_path: Path, thread_id: str) -> None:
    """Remove a thread and all its messages (cascade on FK)."""
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM chat_threads WHERE thread_id = ?", (thread_id,))
        conn.commit()

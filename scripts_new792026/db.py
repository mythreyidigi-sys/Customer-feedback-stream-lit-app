"""
db.py - the single SQLite layer for the Empathy Engine.

Every other module (import pipeline, classifier, analytics, playbook,
replies, monthly report, app.py) imports from here and never opens
sqlite3 connections on its own.
"""

import hashlib
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DB_PATH = Path(os.getenv("EMPATHY_DB", str(Path(__file__).resolve().with_name("empathy.db"))))

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT NOT NULL,          -- google / tripadvisor / zomato
    file_name    TEXT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    rows_in      INTEGER DEFAULT 0,
    rows_added   INTEGER DEFAULT 0,
    status       TEXT DEFAULT 'running', -- running / ok / failed
    error        TEXT
);

CREATE TABLE IF NOT EXISTS reviews (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    review_uid      TEXT NOT NULL UNIQUE,  -- hash, blocks duplicate imports
    source          TEXT NOT NULL,
    branch          TEXT NOT NULL,
    author          TEXT,
    rating          REAL,
    review_date     TEXT,                  -- ISO yyyy-mm-dd
    text            TEXT,
    issue           TEXT,                  -- classifier output
    emotion         TEXT,
    urgency         TEXT,                  -- low / medium / high
    reply_text      TEXT,
    reply_source    TEXT,                  -- groq / template
    reply_status    TEXT DEFAULT 'pending',
    run_id          INTEGER REFERENCES runs(id),
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reviews_branch_date ON reviews(branch, review_date);
CREATE INDEX IF NOT EXISTS idx_reviews_issue       ON reviews(issue);

CREATE TABLE IF NOT EXISTS actions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id   INTEGER REFERENCES reviews(id) ON DELETE CASCADE,
    branch      TEXT,
    issue       TEXT,
    action      TEXT NOT NULL,          -- fix from the playbook
    owner       TEXT,
    status      TEXT DEFAULT 'open',    -- open / in_progress / done
    due_date    TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS reports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    month       TEXT NOT NULL,          -- yyyy-mm
    branch      TEXT NOT NULL DEFAULT 'ALL',
    summary     TEXT,
    pdf_path    TEXT,
    created_at  TEXT NOT NULL,
    UNIQUE (month, branch)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def get_conn():
    """Open a short-lived connection, commit on success, roll back on error."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")  # app can read while imports write
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# ---------- runs ----------

def start_run(source: str, file_name: str | None = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO runs (source, file_name, started_at) VALUES (?, ?, ?)",
            (source, file_name, _now()),
        )
        return cur.lastrowid


def finish_run(run_id: int, rows_in: int, rows_added: int, error: str | None = None) -> None:
    with get_conn() as conn:
        conn.execute(
            """UPDATE runs SET finished_at=?, rows_in=?, rows_added=?, status=?, error=?
               WHERE id=?""",
            (_now(), rows_in, rows_added, "failed" if error else "ok", error, run_id),
        )


# ---------- reviews ----------

def make_uid(source, branch, author, review_date, text) -> str:
    key = "|".join(str(x or "").strip().lower() for x in (source, branch, author, review_date, text))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


REVIEW_COLS = [
    "source", "branch", "author", "rating", "review_date", "text",
    "issue", "emotion", "urgency",
]


def insert_reviews(df: pd.DataFrame, run_id: int) -> int:
    """Insert cleaned + classified reviews. Duplicates are skipped. Returns rows added."""
    df = df.copy()
    for col in REVIEW_COLS:
        if col not in df.columns:
            df[col] = None
    df["review_uid"] = df.apply(
        lambda r: make_uid(r.source, r.branch, r.author, r.review_date, r.text), axis=1
    )
    df["run_id"] = run_id
    df["created_at"] = _now()

    cols = ["review_uid", *REVIEW_COLS, "run_id", "created_at"]
    rows = [tuple(None if pd.isna(v) else v for v in row) for row in df[cols].itertuples(index=False)]
    placeholders = ",".join("?" * len(cols))

    with get_conn() as conn:
        before = conn.total_changes
        conn.executemany(
            f"INSERT OR IGNORE INTO reviews ({','.join(cols)}) VALUES ({placeholders})", rows
        )
        return conn.total_changes - before


def load_reviews(branch: str | None = None, start: str | None = None,
                 end: str | None = None, search: str | None = None) -> pd.DataFrame:
    sql, params = "SELECT * FROM reviews WHERE 1=1", []
    if branch and branch != "ALL":
        sql += " AND branch = ?"; params.append(branch)
    if start:
        sql += " AND review_date >= ?"; params.append(start)
    if end:
        sql += " AND review_date <= ?"; params.append(end)
    if search:
        sql += " AND text LIKE ?"; params.append(f"%{search}%")
    sql += " ORDER BY review_date DESC"
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def save_reply(review_id: int, reply_text: str, reply_source: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE reviews SET reply_text=?, reply_source=?, reply_status='drafted' WHERE id=?",
            (reply_text, reply_source, review_id),
        )


# ---------- actions ----------

def add_action(review_id, branch, issue, action, owner=None, due_date=None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO actions (review_id, branch, issue, action, owner, due_date, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (review_id, branch, issue, action, owner, due_date, _now()),
        )
        return cur.lastrowid


def update_action_status(action_id: int, status: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE actions SET status=?, updated_at=? WHERE id=?",
                     (status, _now(), action_id))


def load_actions(status: str | None = None) -> pd.DataFrame:
    sql, params = "SELECT * FROM actions", []
    if status:
        sql += " WHERE status = ?"; params.append(status)
    sql += " ORDER BY created_at DESC"
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=params)


# ---------- reports ----------

def save_report(month: str, summary: str, pdf_path: str | None, branch: str = "ALL") -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO reports (month, branch, summary, pdf_path, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(month, branch) DO UPDATE SET
                 summary=excluded.summary, pdf_path=excluded.pdf_path,
                 created_at=excluded.created_at""",
            (month, branch, summary, pdf_path, _now()),
        )


def load_reports() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query("SELECT * FROM reports ORDER BY month DESC", conn)


if __name__ == "__main__":
    init_db()
    print(f"Initialised {DB_PATH}")

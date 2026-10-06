"""SQLite storage: reviews, actions, reports, runs (empathy.db)."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from .config import DB_PATH, canonical_chain_name

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    review_hash   TEXT UNIQUE,              -- de-duplication key
    source        TEXT NOT NULL,            -- Google / Zomato / TripAdvisor
    restaurant    TEXT NOT NULL,            -- cleaned chain name
    branch        TEXT,                     -- cleaned branch / locality
    review_date   TEXT NOT NULL,            -- ISO date YYYY-MM-DD
    rating        REAL,                     -- 1-5 (TripAdvisor bubbles / Zomato stars)
    text          TEXT NOT NULL,            -- original review text
    text_clean    TEXT,                     -- cleaned text used for classification
    issue         TEXT,                     -- issue category
    issue_source  TEXT,                     -- 'file' (your HDBSCAN label) or 'classifier'
    sentiment     TEXT,                     -- positive / neutral / negative
    emotion       TEXT,                     -- anger / frustration / disappointment / neutral / delight
    urgency       INTEGER,                  -- 1-10
    red_flag      INTEGER DEFAULT 0,        -- 1 if safety/legal words found
    imported_at   TEXT
);
CREATE INDEX IF NOT EXISTS ix_reviews_rest_date ON reviews(restaurant, review_date);

CREATE TABLE IF NOT EXISTS actions (
    review_id     INTEGER PRIMARY KEY REFERENCES reviews(id),
    status        TEXT DEFAULT 'New',       -- New / In progress / Replied / Resolved
    assignee      TEXT DEFAULT 'Unassigned',
    action_text   TEXT,                     -- recommended / corrective action
    reply_text    TEXT,                     -- approved reply
    notes         TEXT,
    created_at    TEXT,
    replied_at    TEXT,
    resolved_at   TEXT,
    updated_at    TEXT
);

CREATE TABLE IF NOT EXISTS reports (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    restaurant    TEXT,
    month         TEXT,                     -- YYYY-MM
    summary       TEXT,
    pdf_path      TEXT,
    created_at    TEXT,
    UNIQUE(restaurant, month)
);

CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT,
    source        TEXT,
    file_name     TEXT,
    rows_read     INTEGER,
    rows_added    INTEGER,
    rows_skipped  INTEGER,
    message       TEXT
);
"""


def now():
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with connect() as con:
        con.executescript(SCHEMA)


def insert_reviews(df: pd.DataFrame) -> int:
    """Insert classified reviews; duplicates (same review_hash) are skipped. Returns rows added."""
    cols = ["review_hash", "source", "restaurant", "branch", "review_date", "rating", "text", "text_clean",
            "issue", "issue_source", "sentiment", "emotion", "urgency", "red_flag"]
    rows = [tuple(None if pd.isna(v) else v for v in r) + (now(),) for r in df[cols].itertuples(index=False)]
    with connect() as con:
        before = con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        con.executemany(
            f"INSERT OR IGNORE INTO reviews ({','.join(cols)}, imported_at) VALUES ({','.join('?' * (len(cols) + 1))})",
            rows,
        )
        after = con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    return after - before


def log_run(source, file_name, read, added, message=""):
    with connect() as con:
        con.execute(
            "INSERT INTO runs (started_at, source, file_name, rows_read, rows_added, rows_skipped, message) "
            "VALUES (?,?,?,?,?,?,?)",
            (now(), source, file_name, read, added, read - added, message),
        )


def load_reviews() -> pd.DataFrame:
    """All reviews joined with their action status."""
    with connect() as con:
        df = pd.read_sql_query(
            """SELECT r.*, COALESCE(a.status,'New') AS status, COALESCE(a.assignee,'Unassigned') AS assignee,
                      a.action_text, a.reply_text, a.replied_at, a.resolved_at
               FROM reviews r LEFT JOIN actions a ON a.review_id = r.id""",
            con,
        )
    if not df.empty:
        df["review_date"] = pd.to_datetime(df["review_date"])
        df["restaurant"] = df["restaurant"].map(canonical_chain_name)
    return df


def upsert_action(review_id: int, **fields):
    fields["updated_at"] = now()
    with connect() as con:
        exists = con.execute("SELECT 1 FROM actions WHERE review_id=?", (review_id,)).fetchone()
        if exists:
            sets = ", ".join(f"{k}=?" for k in fields)
            con.execute(f"UPDATE actions SET {sets} WHERE review_id=?", (*fields.values(), review_id))
        else:
            fields["created_at"] = fields["updated_at"]
            keys = ["review_id", *fields.keys()]
            con.execute(
                f"INSERT INTO actions ({','.join(keys)}) VALUES ({','.join('?' * len(keys))})",
                (review_id, *fields.values()),
            )


def save_report(restaurant, month, summary, pdf_path):
    with connect() as con:
        con.execute(
            """INSERT INTO reports (restaurant, month, summary, pdf_path, created_at) VALUES (?,?,?,?,?)
               ON CONFLICT(restaurant, month) DO UPDATE SET summary=excluded.summary,
               pdf_path=excluded.pdf_path, created_at=excluded.created_at""",
            (restaurant, month, summary, str(pdf_path), now()),
        )


def load_runs(limit=20) -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql_query(f"SELECT * FROM runs ORDER BY id DESC LIMIT {int(limit)}", con)


def reset_db():
    with connect() as con:
        con.executescript("DROP TABLE IF EXISTS actions; DROP TABLE IF EXISTS reviews; "
                          "DROP TABLE IF EXISTS reports; DROP TABLE IF EXISTS runs;")
    init_db()

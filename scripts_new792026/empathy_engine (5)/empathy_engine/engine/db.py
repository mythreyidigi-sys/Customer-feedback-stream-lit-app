"""SQLite storage: reviews, actions, reports, runs (empathy.db)."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    review_hash   TEXT UNIQUE,              -- de-duplication key
    batch_id      TEXT,                      -- upload/import batch for precise undo
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
    batch_id      TEXT,
    started_at    TEXT,
    source        TEXT,
    file_name     TEXT,
    rows_read     INTEGER,
    rows_added    INTEGER,
    rows_skipped  INTEGER,
    message       TEXT,
    undone_at     TEXT
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
        for table, column, definition in (
            ("reviews", "batch_id", "TEXT"),
            ("runs", "batch_id", "TEXT"),
            ("runs", "undone_at", "TEXT"),
        ):
            columns = {row["name"] for row in con.execute(f"PRAGMA table_info({table})")}
            if column not in columns:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

        legacy_runs = con.execute(
            "SELECT id, started_at, source FROM runs WHERE batch_id IS NULL AND rows_added > 0 ORDER BY id"
        ).fetchall()
        for run in legacy_runs:
            matches = con.execute(
                "SELECT COUNT(*) FROM runs WHERE source=? AND started_at=? AND rows_added > 0",
                (run["source"], run["started_at"]),
            ).fetchone()[0]
            if matches != 1:
                continue
            batch_id = f"legacy-{run['id']}"
            updated = con.execute(
                "UPDATE reviews SET batch_id=? WHERE source=? AND imported_at=? AND batch_id IS NULL",
                (batch_id, run["source"], run["started_at"]),
            )
            if updated.rowcount:
                con.execute("UPDATE runs SET batch_id=? WHERE id=?", (batch_id, run["id"]))


def insert_reviews(df: pd.DataFrame, batch_id: str) -> int:
    """Insert classified reviews; duplicates (same review_hash) are skipped. Returns rows added."""
    cols = ["review_hash", "source", "restaurant", "branch", "review_date", "rating", "text", "text_clean",
            "issue", "issue_source", "sentiment", "emotion", "urgency", "red_flag"]
    rows = [tuple(None if pd.isna(v) else v for v in r) + (batch_id, now())
            for r in df[cols].itertuples(index=False)]
    with connect() as con:
        before = con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        con.executemany(
            f"INSERT OR IGNORE INTO reviews ({','.join(cols)}, batch_id, imported_at) "
            f"VALUES ({','.join('?' * (len(cols) + 2))})",
            rows,
        )
        after = con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    return after - before


def log_run(source, file_name, read, added, message="", batch_id=None):
    with connect() as con:
        con.execute(
            "INSERT INTO runs (batch_id, started_at, source, file_name, rows_read, rows_added, rows_skipped, message) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (batch_id, now(), source, file_name, read, added, read - added, message),
        )


def latest_import_batch():
    with connect() as con:
        return con.execute(
            """SELECT batch_id, SUM(rows_added) AS rows_added, GROUP_CONCAT(file_name, ', ') AS file_names
               FROM runs WHERE batch_id IS NOT NULL AND undone_at IS NULL
               GROUP BY batch_id
               HAVING SUM(rows_added) > 0 AND EXISTS (
                   SELECT 1 FROM reviews WHERE reviews.batch_id = runs.batch_id
               )
               ORDER BY MAX(id) DESC LIMIT 1"""
        ).fetchone()


def undo_import_batch(batch_id: str) -> int:
    with connect() as con:
        con.execute("DELETE FROM actions WHERE review_id IN (SELECT id FROM reviews WHERE batch_id=?)", (batch_id,))
        deleted = con.execute("DELETE FROM reviews WHERE batch_id=?", (batch_id,)).rowcount
        con.execute("UPDATE runs SET undone_at=? WHERE batch_id=? AND undone_at IS NULL", (now(), batch_id))
    return deleted


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

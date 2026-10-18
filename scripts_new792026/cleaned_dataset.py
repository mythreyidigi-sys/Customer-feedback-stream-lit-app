"""Load the already-cleaned review file exactly as it is. No cleaning happens here.

Refresh the data with `python clean_reviews.py All_Reviews_Merged.xlsx All_Reviews_Cleaned.csv`
and update All_Reviews_Cleaned.expected.json with the counts that script prints.
"""
import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("empathy.cleaned_dataset")

CSV_NAME = "All_Reviews_Cleaned.csv"
XLSX_NAME = "All_Reviews_Cleaned.xlsx"
EXPECTED_NAME = "All_Reviews_Cleaned.expected.json"
BATCH_ID = "cleaned_reviews_file"
COLUMNS = ["Review_ID", "Source", "Restaurant", "Branch", "Branch_Original", "Review", "Date", "Tags_Only"]
UNSPECIFIED = "Unspecified"


def find_cleaned_file(folder: Path) -> Path | None:
    for name in (CSV_NAME, XLSX_NAME):
        if (folder / name).is_file():
            return folder / name
    return None


def load_cleaned_reviews(path: Path) -> pd.DataFrame:
    """Read every cell as text, untouched: no dedupe, no row drops, no renames."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    else:
        df = pd.read_excel(path, dtype=str, keep_default_na=False)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    return df


def review_count(df: pd.DataFrame) -> int:
    return len(df)


def branch_count(df: pd.DataFrame) -> int:
    """Distinct (Restaurant, Branch) pairs, ignoring Unspecified."""
    return len(df.loc[df["Branch"] != UNSPECIFIED, ["Restaurant", "Branch"]].drop_duplicates())


def branches_for(df: pd.DataFrame, restaurant: str) -> list[str]:
    scope = df[(df["Restaurant"] == restaurant) & (df["Branch"] != UNSPECIFIED)]
    return sorted(scope["Branch"].unique())


def check_counts(df: pd.DataFrame, folder: Path) -> list[str]:
    """Warn when the loaded counts differ from the ones clean_reviews.py printed."""
    expected_path = Path(folder) / EXPECTED_NAME
    if not expected_path.is_file():
        return []
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    actual = {"rows": review_count(df), "branches": branch_count(df)}
    warnings = []
    for key, label in (("rows", "review count"), ("branches", "branch count")):
        if key in expected and int(expected[key]) != actual[key]:
            message = (f"Cleaned reviews {label} mismatch: loaded {actual[key]}, "
                       f"clean_reviews.py reported {expected[key]}")
            log.warning(message)
            warnings.append(message)
    return warnings


def to_app_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map the cleaned columns to the app's review columns without changing any value."""
    return pd.DataFrame({
        "review_hash": df["Review_ID"],
        "source": df["Source"],
        "restaurant": df["Restaurant"],
        "branch": df["Branch"],
        "review_date": df["Date"],
        "rating": None,
        "text": df["Review"],
        "text_clean": df["Review"].str.lower(),  # the classifier's keyword rules expect lower case
    })


def batch_id_for(path: Path) -> str:
    """One batch per version of the file, so a refreshed file loads again but an undone one does not."""
    return f"{BATCH_ID}_{int(Path(path).stat().st_mtime)}"


def sync_to_database(df: pd.DataFrame, db, classify, batch_id: str, file_name: str) -> int:
    """Load the cleaned file once as an undoable batch. Returns rows added.

    A batch that was already loaded, or undone by the user, is never loaded again.
    """
    db.init_db()
    with db.connect() as con:
        seen = con.execute("SELECT 1 FROM runs WHERE batch_id=? LIMIT 1", (batch_id,)).fetchone()
        old_batches = [r[0] for r in con.execute(
            "SELECT DISTINCT batch_id FROM runs WHERE batch_id LIKE ? AND batch_id<>? AND undone_at IS NULL",
            (f"{BATCH_ID}_%", batch_id))]
    if seen:
        return 0
    for old_batch in [*old_batches, BATCH_ID]:  # BATCH_ID: rows loaded before batches were logged
        db.undo_import_batch(old_batch)
    added = db.insert_reviews(classify(to_app_frame(df)), batch_id)
    db.log_run("Cleaned file", file_name, len(df), added, "Loaded from the cleaned review file", batch_id)
    return added

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


def sync_to_database(df: pd.DataFrame, db, classify) -> int:
    """Make the app database hold exactly the rows of the cleaned file. Returns rows added."""
    ids = set(df["Review_ID"])
    db.init_db()
    with db.connect() as con:
        existing = {r[0] for r in con.execute(
            "SELECT review_hash FROM reviews WHERE batch_id=?", (BATCH_ID,))}
        stale = existing - ids
        for review_hash in stale:
            con.execute("DELETE FROM actions WHERE review_id IN (SELECT id FROM reviews WHERE review_hash=?)",
                        (review_hash,))
            con.execute("DELETE FROM reviews WHERE review_hash=?", (review_hash,))
    new_rows = df[~df["Review_ID"].isin(existing)]
    if new_rows.empty:
        return 0
    return db.insert_reviews(classify(to_app_frame(new_rows)), BATCH_ID)

"""Import pipeline: read Google / Zomato / TripAdvisor CSV or Excel files, clean them, classify, store.

Handles the usual differences between scraped files:
  * different column names (review_text / Review / content ...)
  * restaurant + branch in one field ("Geetham - T. Nagar", "A2B Adyar Ananda Bhavan, Velachery")
  * dates as ISO, dd/mm/yyyy, "14 May 2026", "May 2026" or relative ("2 months ago")
  * ratings as 4, "4.0 stars", "Rated 4.0" or TripAdvisor bubbles (40 -> 4)
"""
import hashlib
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pandas as pd

from . import db
from .classifier import classify
from .config import CHAIN_ALIASES, RAW_DIR, SOURCES

COLUMN_SYNONYMS = {
    "text": ["review_text", "text", "review", "reviews", "comment", "content", "body",
             "description", "review_body", "snippet", "cleaned_review", "clean_text"],
    "rating": ["rating", "rated", "stars", "star", "score", "rating_value", "review_rating", "bubble", "bubbles"],
    "date": ["review_date", "date", "published", "published_at", "posted", "posted_on", "time",
             "relative_date", "review_time", "date_of_review", "reviewed_on", "timestamp", "visit_date"],
    "restaurant": ["restaurant", "restaurant_name", "chain", "brand", "place",
                   "place_name", "hotel", "outlet_name", "business", "name"],
    "branch": ["branch", "branch_name", "location", "locality", "area", "outlet", "address", "city_area"],
    "source": ["source", "platform", "site"],
    "issue": ["issue", "issue_category", "issue_label", "cluster_label", "cluster_name", "category", "label"],
}
MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
EMOJI_RX = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF‍️]+")


# ----------------------------------------------------------------------------- reading
def read_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {path.name}")


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower = {re.sub(r"[\s-]+", "_", str(c).lower().strip()): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for target, options in COLUMN_SYNONYMS.items():
        for opt in options:
            if opt in lower:
                out[target] = df[lower[opt]]
                break
    if "text" not in out:
        raise ValueError(f"No review text column found. Columns in file: {list(df.columns)}")
    return out


def source_from_filename(file_name: str) -> str | None:
    name = re.sub(r"[_-]+", " ", file_name.lower())
    if "zomato" in name:
        return "Zomato"
    if "tripadvisor" in name or "trip advisor" in name:
        return "TripAdvisor"
    if "google" in name:
        return "Google"
    return None


# ----------------------------------------------------------------------------- cleaning
def clean_text(t) -> str:
    if not isinstance(t, str):
        return ""
    t = t.replace("\n", " ")
    if "(Translated by Google)" in t:  # keep the English part of Google translations
        t = t.split("(Translated by Google)", 1)[1].split("(Original)", 1)[0]
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = EMOJI_RX.sub(" ", t)
    t = re.sub(r"[^A-Za-z0-9.,!?'\s-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def display_text(t) -> str:
    if not isinstance(t, str):
        return ""
    if "(Translated by Google)" in t:
        t = t.split("(Translated by Google)", 1)[1].split("(Original)", 1)[0]
    return re.sub(r"\s+", " ", t).strip()


def split_name(raw_name, raw_branch):
    """Returns (chain, branch)."""
    name = str(raw_name or "").strip()
    low = name.lower()
    chain, rest = None, ""
    for alias in sorted(CHAIN_ALIASES, key=len, reverse=True):
        if low.startswith(alias) or re.search(rf"\b{re.escape(alias)}\b", low):
            chain = CHAIN_ALIASES[alias]
            rest = re.sub(re.escape(alias), "", name, count=1, flags=re.I)
            break
    if chain is None:
        chain = re.split(r"\s[-|–]\s|,|\(", name)[0].strip().title() or "Unknown"
        rest = name[len(chain):]
    branch = raw_branch if isinstance(raw_branch, str) and raw_branch.strip() else rest
    return chain, clean_branch(branch)


def clean_branch(b) -> str:
    b = str(b or "")
    b = re.sub(r"(?i)\b(chennai|tamil nadu|india|restaurant|veg|pure veg|hotel|branch)\b", " ", b)
    b = re.sub(r"\b\d{6}\b|\d+(st|nd|rd|th)?\s*(floor|main road)?", " ", b, flags=re.I)
    parts = [p.strip(" -|()–.") for p in re.split(r"[,(|)]| - ", b) if p.strip(" -|()–.")]
    if not parts:
        return "Main"
    # an address -> the locality is usually the last meaningful part
    part = parts[-1] if len(parts) > 2 else parts[0]
    part = re.sub(r"\s+", " ", part).strip()
    part = part.title() if part.islower() or part.isupper() and len(part) > 4 else part
    return part.replace("T Nagar", "T. Nagar") or "Main"


def parse_rating(v):
    if pd.isna(v):
        return None
    m = re.search(r"\d+(\.\d+)?", str(v))
    if not m:
        return None
    r = float(m.group())
    if r > 5:
        r = r / 10 if r <= 50 else None  # TripAdvisor bubble_40 -> 4.0
    return r if r is not None and 1 <= r <= 5 else None


def parse_date(v, ref: date):
    if pd.isna(v):
        return None
    if isinstance(v, (datetime, pd.Timestamp)):
        return v.date()
    s = str(v).strip().lower()
    s = re.sub(r"^(edited|reviewed|written|date of visit:?|visited)\s*", "", s).strip()
    # relative dates (Google): "2 months ago", "a week ago", "yesterday"
    if "ago" in s or s in ("yesterday", "today"):
        if s == "today":
            return ref
        if s == "yesterday":
            return ref - timedelta(days=1)
        m = re.search(r"(\d+|a|an|one)\s+(minute|hour|day|week|month|year)", s)
        if m:
            n = 1 if m.group(1) in ("a", "an", "one") else int(m.group(1))
            days = {"minute": 0, "hour": 0, "day": 1, "week": 7, "month": 30, "year": 365}[m.group(2)] * n
            return ref - timedelta(days=days)
    # "May 2026" (TripAdvisor date of visit) -> 15th of that month
    if re.fullmatch(rf"({MONTHS})[a-z]*[ ,]+\d{{4}}", s):
        return pd.to_datetime("15 " + s).date()
    dayfirst = bool(re.match(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", s))  # Indian dd/mm/yyyy
    try:
        return pd.to_datetime(s, dayfirst=dayfirst).date()
    except (ValueError, TypeError):
        return None


def normalise(df: pd.DataFrame, source: str, ref: date, force_source: bool = False) -> pd.DataFrame:
    m = map_columns(df)
    out = pd.DataFrame(index=m.index)
    out["source"] = source if force_source or "source" not in m else m["source"].fillna(source).astype(str).str.title()
    out["source"] = out["source"].replace({"Tripadvisor": "TripAdvisor", "Google Reviews": "Google"})
    names = m.get("restaurant", pd.Series("", index=m.index))
    branches = m.get("branch", pd.Series(None, index=m.index))
    split = [split_name(n, b) for n, b in zip(names, branches)]
    out["restaurant"] = [c for c, _ in split]
    out["branch"] = [b for _, b in split]
    out["review_date"] = [parse_date(v, ref) or ref for v in m.get("date", pd.Series(ref, index=m.index))]
    out["rating"] = [parse_rating(v) for v in m.get("rating", pd.Series(None, index=m.index))]
    out["text"] = m["text"].apply(display_text)
    out["text_clean"] = m["text"].apply(clean_text)
    out["issue"] = m["issue"] if "issue" in m else None

    before = len(out)
    out = out[(out["text_clean"].str.len() >= 3) & out["review_date"].notna()]
    out["review_date"] = pd.to_datetime(out["review_date"]).dt.strftime("%Y-%m-%d")
    out["review_hash"] = [
        hashlib.md5(f"{s}|{r}|{b}|{d}|{t}".encode()).hexdigest()
        for s, r, b, d, t in zip(out["source"], out["restaurant"], out["branch"], out["review_date"], out["text_clean"])
    ]
    out = out.drop_duplicates("review_hash")
    out.attrs["dropped"] = before - len(out)
    return out


# ----------------------------------------------------------------------------- entry points
def import_file(path, source: str, ref: date | None = None, batch_id: str | None = None,
                force_source: bool = False) -> dict:
    """Import one file. `ref` = date the data was scraped (used for '2 months ago' style dates)."""
    path = Path(path)
    ref = ref or date.fromtimestamp(path.stat().st_mtime)
    batch_id = batch_id or uuid4().hex
    db.init_db()
    try:
        raw = read_file(path)
        clean = normalise(raw, source, ref, force_source=force_source)
        classified = classify(clean)
        added = db.insert_reviews(classified, batch_id)
        msg = f"{clean.attrs.get('dropped', 0)} rows dropped (empty text / no date / duplicate)"
        db.log_run(source, path.name, len(raw), added, msg, batch_id)
        return {"file": path.name, "source": source, "read": len(raw), "added": added, "message": msg}
    except Exception as e:  # keep going with the other files
        db.log_run(source, path.name, 0, 0, f"ERROR: {e}", batch_id)
        return {"file": path.name, "source": source, "read": 0, "added": 0, "message": f"ERROR: {e}"}


def import_folder(raw_dir: Path = RAW_DIR, ref: date | None = None) -> list[dict]:
    """Import every CSV/Excel in data/raw/google, data/raw/zomato, data/raw/tripadvisor."""
    results = []
    for source in SOURCES:
        folder = Path(raw_dir) / source.lower()
        for f in sorted(folder.glob("*")):
            if f.suffix.lower() in (".csv", ".xlsx", ".xls"):
                results.append(import_file(f, source_from_filename(f.name) or source, ref))
    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Import review files into empathy.db")
    p.add_argument("--scraped-on", help="YYYY-MM-DD the files were scraped (for relative dates)")
    p.add_argument("--reset", action="store_true", help="wipe the database first")
    a = p.parse_args()
    if a.reset:
        db.reset_db()
    ref = date.fromisoformat(a.scraped_on) if a.scraped_on else None
    for r in import_folder(ref=ref):
        print(f"{r['source']:<12} {r['file']:<40} read={r['read']:<6} added={r['added']:<6} {r['message']}")

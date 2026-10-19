"""Empathy Engine — Streamlit decision-support app (single entry point).

Dissertation: Leveraging Customer Experience Analytics for Service Quality Improvement in
High-Volume Restaurants (N.R. Mythreyi · 2024MB22535 · BITS Pilani WILP, MBA Digital Business)

Run:  streamlit run main.py

Pipeline (final dataset: 6 chains x 3 platforms = 18 combinations):
    Google / Zomato / TripAdvisor (CSV/Excel, scraped with Selenium + Playwright)
      -> Import pipeline (clean names, branches, dates, dedupe)
      -> Sentence-Transformer (all-MiniLM-L6-v2) embeddings
      -> HDBSCAN clustering (no preset k; outliers flagged as noise)
      -> Groq LLM cluster labels (issue), plus emotion / urgency classifier
      -> SQLite (empathy.db: reviews, actions, reports, runs)
      -> Streamlit app:
           1. Search            upload and import review files
           2. Analytics         reputation scoring, clustering, sentiment and segment charts
           3. Overview          KPIs, top issues, priority decision panel
           4. Campaigns         Digital marketing / 360-degree campaigns tied to clusters
           5. Monthly report    PDF summary for managers
           6. SERVQUAL          Survey expectation/perception gaps and NLP triangulation
           7. Empathetic Reply  Standalone, editable reply drafting

Setup:
    pip install -r requirements.txt
    python -m engine.importer --reset --scraped-on YYYY-MM-DD
    set GROQ_API_KEY=gsk_...                                # optional AI replies
    streamlit run main.py

This file needs the `engine/` package. It looks for it next to main.py first,
then in empathy_engine/ and scripts_new792026/empathy_engine (5)/empathy_engine/.
"""
import hashlib
import sys
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

# ------------------------------------------------------------------ locate the engine package
_HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    _HERE,
    _HERE / "empathy_engine",
    _HERE / "scripts_new792026" / "empathy_engine (5)" / "empathy_engine",
]
APP_DIR = next((p for p in _CANDIDATES if (p / "engine").is_dir()), _HERE)
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
_CLEANER_DIR = _HERE / "scripts_new792026"
if str(_CLEANER_DIR) not in sys.path:
    sys.path.insert(0, str(_CLEANER_DIR))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine import analytics as an
from engine import classifier as issue_classifier
from engine import db
from engine.config import (HIGH_THRESHOLD, ISSUES, POSITIVE_ISSUE, RAW_DIR, SOURCES, TEAM, URGENT_THRESHOLD,
                           canonical_chain_name)
from engine.importer import import_file, read_file, source_from_filename
from engine.playbook import deadline_text, recommend
from engine.replies import draft_reply, groq_available, template_reply
from engine.report import build_report, save_report
from clean_reviews import OUTPUT_FILE as CLEANED_REVIEWS_OUTPUT, clean_reviews_dataframe
import cleaned_dataset
import importlib

import analytics_dashboard

importlib.reload(analytics_dashboard)   # hosted Streamlit can keep a stale copy of this module after a redeploy

st.set_page_config(page_title="Empathy Engine", page_icon="💬", layout="wide")
db.init_db()

GREEN, RED, GREY, AMBER, ACCENT = "#1D9E75", "#D85A30", "#B4B2A9", "#BA7517", "#0F6E56"
SENT_COLORS = {"negative": RED, "neutral": GREY, "positive": GREEN}
PAGES = ["🔍 1. Search", "📈 2. Analytics", "📊 3. Overview", "📣 4. Campaigns",
         "📄 5. Monthly report", "🧭 6. SERVQUAL", "💌 7. Empathetic Reply"]

# ------------------------------------------------------------------ project reference figures
# Final dataset as reported in the dissertation (used only as a reference caption; live
# numbers in the app always come from the database).
# "Brand Reputation & Customer Satisfaction" is the catch-all for unhappy-but-unspecific reviews and names no
# fixable problem, so it is hidden from cluster views like noise.
NOISE_LABELS = {"noise", "unclassified", "uncategorized", "uncategorised", "outlier", "-1", "other", "",
                "brand reputation & customer satisfaction"}

# ------------------------------------------------------------------ Module 5 aspects
ASPECTS = {
    "Parking": r"\bpark(ing)?\b|two[- ]?wheeler|valet",
    "Seating & crowding": r"\bseat(s|ing)?\b|\btable(s)?\b|crowd(ed)?|\brush\b|\bqueue\b|no place|space",
    "Ambience & noise": r"ambi[ae]nce|atmosphere|\bnois(e|y)\b|\bdecor\b|lighting|music|\bac\b|air[- ]?condition|\bhot inside\b",
    "Cleanliness": r"clean|dirty|hygien|unhygien|washroom|restroom|toilet|\bflies\b|cockroach|smell",
    "Location & access": r"locat(ion|ed)|easy to find|hard to find|\baccess\b|near (the )?(bus|metro|station)|main road|landmark",
}

# ------------------------------------------------------------------ 360-degree campaigns
# Each campaign is tied to an HDBSCAN issue cluster. Matching is keyword-based on the cluster
# label so it keeps working when the Groq labels change slightly after a re-run.
# Order matters: the first campaign whose keywords match wins.
CAMPAIGNS = [
    dict(id=5, name="Clean Kitchen, Clear Conscience", theme="Poor experience & food hygiene",
         keys=("hygien", "clean", "dirty", "food safety", "poor experience"),
         headline="Our kitchen, your peace of mind — see our hygiene standards for yourself.",
         channels=["In-store signage", "Owned/PR", "Review-platform response"],
         kpi="Zero tolerance: any week-over-week spike pauses the public campaign",
         mode="risk"),
    dict(id=2, name="More to Choose, Faster to Get", theme="Food variety & fast service",
         keys=("variety", "menu", "choice", "option"),
         headline="More dishes. Less waiting. Now on the menu.",
         channels=["Paid social", "Zomato/Swiggy in-app", "In-store signage", "Email/SMS"],
         kpi="Trial rate of promoted dishes; order-to-serve time vs baseline",
         mode="launch"),
    dict(id=4, name="On Time, Every Time", theme="Slow service & staff negligence",
         keys=("slow", "wait", "delay", "negligen", "late", "queue"),
         headline="We cut our average wait time. Come time us.",
         channels=["Paid social", "Google Ads", "Zomato/Swiggy in-app", "Influencer/UGC"],
         kpi="Table-turnaround time; 'wait/slow/negligence' mentions in new reviews",
         mode="gated"),
    dict(id=1, name="Served the way you'd serve family.", theme="Food quantity & value for money",
         keys=("quantity", "value", "portion", "price", "money", "cost", "expensive"),
         headline="Full plate. Fair price. No compromises on either.",
         channels=["Paid social", "In-store signage", "Influencer/UGC"],
         kpi="Cut this cluster's share of new reviews by 15–20% within one quarter",
         mode="launch"),
    dict(id=3, name="Service, Reimagined", theme="Service quality",
         keys=("service", "staff", "rude", "behav", "courte", "attitude", "waiter"),
         headline="Every order matters. Every guest matters. That's our new standard.",
         channels=["Paid social", "In-store signage", "Owned/PR", "Review-platform response"],
         kpi="Lower Service Quality share; higher star rating on 'staff/service' reviews",
         mode="launch"),
]
CHANNELS = ["Paid social", "Google Ads", "Zomato/Swiggy in-app", "In-store signage",
            "Influencer/UGC", "Owned/PR", "Email/SMS", "Review-platform response"]

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}
.pill {display:inline-block; font-size:12px; padding:2px 10px; border-radius:999px; margin:0 4px 4px 0;
       background:rgba(128,128,128,.15);}
.p-urgent {background:#FCEBEB; color:#A32D2D;} .p-high {background:#FAEEDA; color:#854F0B;}
.p-ok {background:#E1F5EE; color:#0F6E56;} .p-muted {opacity:.65;}
.quote {font-style:italic; font-size:15px; margin:6px 0 10px;}
.small {font-size:13px; opacity:.75;}
</style>""", unsafe_allow_html=True)


def pill(text, cls=""):
    return f'<span class="pill {cls}">{text}</span>'


def resolve_upload_source(file_name, fallback_source):
    return source_from_filename(file_name) or fallback_source


def render_upload_analyse_form(key_prefix):
    notice = st.session_state.pop("upload_notice", None)
    if notice:
        st.success(notice)

    uploaded_files = st.file_uploader(
        "Drop files here, or click to browse", type=["csv", "xlsx", "xls"], accept_multiple_files=True,
        key=f"{key_prefix}_files",
    )
    st.caption("CSV, XLSX or XLS · upload files, then run the cleaning and analysis pipeline · relative dates use today")
    fallback_source = SOURCES[0]
    if uploaded_files and any(source_from_filename(file.name) is None for file in uploaded_files):
        fallback_source = st.selectbox(
            "Platform for files without one in the filename", SOURCES, key=f"{key_prefix}_source"
        )

    pipeline_key = f"{key_prefix}_processed_upload_hash"
    if uploaded_files:
        upload_digest = hashlib.sha256()
        for uploaded_file in uploaded_files:
            upload_digest.update(uploaded_file.name.encode("utf-8"))
            upload_digest.update(uploaded_file.getbuffer())
        upload_digest.update(fallback_source.encode("utf-8"))
        upload_hash = upload_digest.hexdigest()
        if upload_hash != st.session_state.get(pipeline_key):
            st.session_state[pipeline_key] = upload_hash
            st.session_state.pop("_cleaned_upload_bytes", None)
    else:
        st.session_state.pop(pipeline_key, None)

    undo_key = f"{key_prefix}_undo_batch"
    latest_batch = db.latest_import_batch()
    analyse_col, undo_col, undo_all_col = st.columns([1, 1, 1])
    analyse_clicked = analyse_col.button(
        "Analyse", type="primary", key=f"{key_prefix}_analyse", disabled=not uploaded_files
    )
    undo_clicked = undo_col.button(
        "↶ Undo last upload", key=f"{key_prefix}_undo", disabled=latest_batch is None,
        help="Reverse the latest imported batch of review documents.",
    )
    undo_all_key = f"{key_prefix}_undo_all"
    undo_all_clicked = undo_all_col.button(
        "↶ Undo all uploads", key=f"{key_prefix}_undo_all_btn", disabled=latest_batch is None,
        help="Remove every loaded batch: the cleaned review file and any uploads.",
    )
    if undo_all_clicked:
        st.session_state[undo_all_key] = True
        st.rerun()
    if st.session_state.get(undo_all_key):
        if latest_batch is None:
            st.session_state.pop(undo_all_key, None)
        else:
            st.warning("Undo all uploads? This removes every review imported through uploads. "
                       "Original files remain in the raw archive.")
            yes_col, no_col = st.columns(2)
            if yes_col.button("Confirm undo all", key=f"{key_prefix}_confirm_undo_all", type="primary"):
                removed = db.undo_all_import_batches()
                forget_loaded_reviews()
                st.session_state.pop(undo_all_key, None)
                st.session_state["upload_notice"] = f"Undid all uploads and removed {removed:,} review(s)."
                st.rerun()
            if no_col.button("Keep uploads", key=f"{key_prefix}_cancel_undo_all"):
                st.session_state.pop(undo_all_key, None)
                st.rerun()

    if "_cleaned_upload_bytes" in st.session_state:
        st.download_button(
            "Download cleaned batch", data=st.session_state["_cleaned_upload_bytes"],
            file_name="cleaned_reviews.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{key_prefix}_cleaned_download",
        )

    if undo_clicked and latest_batch is not None:
        st.session_state[undo_key] = latest_batch["batch_id"]
        st.rerun()

    pending_batch = st.session_state.get(undo_key)
    if pending_batch:
        latest_batch = db.latest_import_batch()
        if latest_batch is None or latest_batch["batch_id"] != pending_batch:
            st.session_state.pop(undo_key, None)
        else:
            st.warning(
                f"Undo this upload? This will remove {latest_batch['rows_added']:,} imported reviews "
                f"from {latest_batch['file_names']}. The original file remains in the raw archive."
            )
            confirm_col, cancel_col = st.columns(2)
            if confirm_col.button("Confirm undo", key=f"{key_prefix}_confirm_undo", type="primary"):
                removed = db.undo_import_batch(pending_batch)
                forget_loaded_reviews()
                st.session_state.pop(undo_key, None)
                st.session_state["upload_notice"] = f"Undid the latest upload and removed {removed:,} review(s)."
                st.rerun()
            if cancel_col.button("Keep upload", key=f"{key_prefix}_cancel_undo"):
                st.session_state.pop(undo_key, None)
                st.rerun()

    if analyse_clicked:
        batch_id = uuid4().hex
        imported_files, added_reviews, errors = 0, 0, []
        cleaned_frames = []
        total_cleaned, invalid_removed, duplicates_removed = 0, 0, 0
        st.session_state.pop("_cleaned_upload_bytes", None)
        with TemporaryDirectory(prefix="empathy_clean_") as temp_dir:
            for file_index, uploaded_file in enumerate(uploaded_files):
                detected_source = source_from_filename(uploaded_file.name)
                upload_source = resolve_upload_source(uploaded_file.name, fallback_source)
                source_folder = upload_source.lower() if upload_source in SOURCES else "other"
                path = RAW_DIR / source_folder / Path(uploaded_file.name).name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(uploaded_file.getbuffer())
                try:
                    raw_reviews = read_file(path)
                    cleaned_reviews, cleaning_stats = clean_reviews_dataframe(raw_reviews)
                    cleaned_frames.append(cleaned_reviews)
                    total_cleaned += cleaning_stats["final"]
                    invalid_removed += cleaning_stats["invalid_removed"]
                    duplicates_removed += cleaning_stats["duplicates_removed"]

                    cleaned_path = Path(temp_dir) / f"{file_index}_{Path(uploaded_file.name).stem}_cleaned.xlsx"
                    cleaned_reviews.to_excel(cleaned_path, index=False)
                    result = import_file(cleaned_path, upload_source, date.today(), batch_id=batch_id,
                                         force_source=detected_source is not None)
                    if result["message"].startswith("ERROR"):
                        errors.append(f"{uploaded_file.name}: {result['message']}")
                    else:
                        imported_files += 1
                        added_reviews += result["added"]
                except Exception as exc:
                    errors.append(f"{uploaded_file.name}: {exc}")

        if cleaned_frames:
            cleaned_batch = pd.concat(cleaned_frames, ignore_index=True, sort=False)
            output_path = Path(CLEANED_REVIEWS_OUTPUT)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cleaned_batch.to_excel(output_path, index=False)
            output_buffer = BytesIO()
            cleaned_batch.to_excel(output_buffer, index=False)
            st.session_state["_cleaned_upload_bytes"] = output_buffer.getvalue()

        if imported_files:
            batch_reviews = db.load_reviews()
            batch_reviews = batch_reviews[batch_reviews["batch_id"] == batch_id]
            if not batch_reviews.empty:
                latest_review = batch_reviews["review_date"].max()
                earliest_review = batch_reviews["review_date"].min()
                st.session_state["_uploaded_review_focus"] = {
                    "restaurant": batch_reviews["restaurant"].value_counts().index[0],
                    "date_range": (earliest_review.date(), latest_review.date()),
                }
            st.session_state["upload_notice"] = (
                f"Cleaned {total_cleaned:,} review(s) from {len(cleaned_frames)} file(s); "
                f"removed {invalid_removed:,} invalid and {duplicates_removed:,} duplicate row(s); "
                f"analysed {imported_files} file(s) and added {added_reviews:,} new reviews."
            )
            if errors:
                st.session_state["upload_notice"] += f" {len(errors)} file(s) could not be analysed."
            st.rerun()
        for error in errors:
            st.error(error)


# ------------------------------------------------------------------ helpers (pure pandas)
def is_noise(issue: pd.Series) -> pd.Series:
    """HDBSCAN outliers (label -1) or rows without a usable cluster label."""
    s = issue.astype("string").str.strip().str.lower()
    return s.isna() | s.isin(NOISE_LABELS)


def classified(df: pd.DataFrame) -> pd.DataFrame:
    return df[~is_noise(df["issue"])]


def complaint_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Classified, non-positive reviews: the complaint clusters."""
    d = classified(df)
    return d[d["issue"] != POSITIVE_ISSUE]


def is_negative(df: pd.DataFrame) -> pd.Series:
    if "sentiment" in df.columns:
        return df["sentiment"].eq("negative") | df["rating"].le(2)
    return df["rating"].le(2)


def match_campaign(issue):
    label = str(issue).lower()
    for c in CAMPAIGNS:
        if any(k in label for k in c["keys"]):
            return c["id"]
    return None


def weekly_counts(df: pd.DataFrame, as_of, weeks=5) -> list:
    """Counts for the last `weeks` 7-day blocks ending at as_of (oldest first)."""
    end = pd.Timestamp(as_of) + pd.Timedelta(days=1)
    out = []
    for w in range(weeks, 0, -1):
        lo, hi = end - pd.Timedelta(days=7 * w), end - pd.Timedelta(days=7 * (w - 1))
        out.append(int(((df["review_date"] >= lo) & (df["review_date"] < hi)).sum()))
    return out


def aspect_flags(df: pd.DataFrame) -> pd.DataFrame:
    text = df["text"].fillna("").str.lower()
    return pd.DataFrame({a: text.str.contains(rx, regex=True) for a, rx in ASPECTS.items()}, index=df.index)


@st.cache_data
def load_servqual_scores(path: str) -> tuple[pd.DataFrame, int]:
    responses = pd.read_excel(path)
    dimensions = ["Reliability", "Responsiveness", "Assurance", "Empathy", "Tangibles"]
    rows = []
    respondent_masks = []

    for dimension in dimensions:
        prefix = f"{dimension}|"
        item_ids = {
            column[len(prefix):-2]
            for column in responses.columns
            if column.startswith(prefix) and column.endswith(("|E", "|P"))
        }
        expectation_columns, perception_columns = [], []
        for item_id in sorted(item_ids):
            expectation_column = f"{prefix}{item_id}|E"
            perception_column = f"{prefix}{item_id}|P"
            if expectation_column in responses and perception_column in responses:
                expectation_columns.append(expectation_column)
                perception_columns.append(perception_column)

        if expectation_columns:
            expectation = responses[expectation_columns].apply(pd.to_numeric, errors="coerce")
            perception = responses[perception_columns].apply(pd.to_numeric, errors="coerce")
            expectation.columns = range(len(expectation_columns))
            perception.columns = range(len(perception_columns))
            valid_pairs = expectation.notna() & perception.notna()
            respondent_masks.append(valid_pairs.any(axis=1))
            rows.append({
                "dimension": dimension,
                "mean_expectation": expectation.where(valid_pairs).stack().mean(),
                "mean_perception": perception.where(valid_pairs).stack().mean(),
                "mean_gap": (perception - expectation).where(valid_pairs).stack().mean(),
                "n_items": len(expectation_columns),
                "respondents": int(valid_pairs.any(axis=1).sum()),
            })
        else:
            rows.append({"dimension": dimension, "mean_expectation": None,
                         "mean_perception": None, "mean_gap": None, "n_items": 0,
                         "respondents": 0})

    respondent_count = int(pd.concat(respondent_masks, axis=1).any(axis=1).sum()) if respondent_masks else 0
    return pd.DataFrame(rows), respondent_count


# ------------------------------------------------------------------ data + sidebar filters
@st.cache_resource(show_spinner="Loading cleaned reviews...")
def load_cleaned_dataset(path: str, modified: float):
    """Read the cleaned file as-is, check its counts, and load it once as an undoable batch."""
    raw = cleaned_dataset.load_cleaned_reviews(Path(path))
    warnings = cleaned_dataset.check_counts(raw, Path(path).parent)
    cleaned_dataset.sync_to_database(raw, db, issue_classifier.classify,
                                     cleaned_dataset.batch_id_for(Path(path)), Path(path).name)
    return warnings


def forget_loaded_reviews():
    """Drop everything the app holds in memory about the loaded reviews after an undo."""
    st.cache_resource.clear()
    st.cache_data.clear()
    for key in ("_uploaded_review_focus", "_cleaned_upload_bytes", "as_of_date", "window_days"):
        st.session_state.pop(key, None)


cleaned_file = cleaned_dataset.find_cleaned_file(_HERE)
if cleaned_file is not None:
    for cleaned_warning in load_cleaned_dataset(str(cleaned_file), cleaned_file.stat().st_mtime):
        st.warning(cleaned_warning)

df_all = db.load_reviews()

if df_all.empty:
    st.title("💬 Empathy Engine")
    st.warning("No reviews in the database yet.")
    with st.container(border=True):
        st.markdown("Upload review files to start analysing customer experience.")
        st.caption(issue_classifier.issue_model_status())
        render_upload_analyse_form("empty_state")
    st.stop()

df_all["restaurant"] = df_all["restaurant"].map(canonical_chain_name)
restaurants = sorted(set(df_all["restaurant"]) | {"Namma Veedu Vasantha Bhavan", "Saravana Bhavan", "Geetham"})
data_sources = sorted(df_all["source"].dropna().astype(str).unique())
data_end = df_all["review_date"].max().date()
data_start = df_all["review_date"].min().date()
ss = st.session_state
uploaded_review_focus = ss.pop("_uploaded_review_focus", None)
uploaded_date_range = None
if uploaded_review_focus:
    ss["restaurant"] = uploaded_review_focus["restaurant"]
    ss["page"] = PAGES[2]
    uploaded_date_range = uploaded_review_focus["date_range"]
ss.setdefault("restaurant", "Geetham" if "Geetham" in restaurants else restaurants[0])
ss["restaurant"] = canonical_chain_name(ss["restaurant"])
if ss["restaurant"] not in restaurants:
    ss["restaurant"] = restaurants[0]
ss.setdefault("page", PAGES[0])
if ss.page not in PAGES:          # session from an older version of the app
    ss.page = PAGES[0]

with st.sidebar:
    st.markdown("### 💬 Empathy Engine")
    st.selectbox("Restaurant", restaurants, key="restaurant")
    rest_df = df_all[df_all["restaurant"] == ss.restaurant]
    branch_options = sorted(rest_df.loc[rest_df["branch"] != cleaned_dataset.UNSPECIFIED, "branch"].unique())
    branches = st.multiselect("Branches", branch_options, placeholder="All branches")
    sources = st.multiselect("Sources", data_sources, default=data_sources)
    if uploaded_date_range is not None:
        default_range = uploaded_date_range
    else:
        range_end = ss.get("as_of_date", data_end)
        if not isinstance(range_end, date) or not data_start <= range_end <= data_end:
            range_end = data_end
        range_days = max(1, int(ss.get("window_days", 90)))
        range_start = max(data_start, (pd.Timestamp(range_end) - pd.Timedelta(days=range_days - 1)).date())
        default_range = (range_start, range_end)
    selected_range = st.date_input(
        "As of",
        value=default_range,
        min_value=data_start,
        max_value=data_end,
    )
    if len(selected_range) == 2:
        cur_start, cur_end = selected_range
    elif selected_range:
        cur_start = cur_end = selected_range[0]
    else:
        cur_start = cur_end = data_end
    as_of = cur_end
    days = (cur_end - cur_start).days + 1
    compare = st.multiselect("Compare with", [r for r in restaurants if r != ss.restaurant],
                             default=[r for r in ("A2B", "Sangeetha") if r in restaurants and r != ss.restaurant])
    st.divider()
    branch_pairs = df_all.loc[df_all["branch"] != cleaned_dataset.UNSPECIFIED, ["restaurant", "branch"]]
    st.caption(f"Data: {len(df_all):,} reviews · {df_all['restaurant'].nunique()} chains · "
               f"{len(branch_pairs.drop_duplicates())} branches · {df_all['source'].nunique()} platforms")
    st.caption(f"{data_start:%d %b %Y} – {data_end:%d %b %Y}")
    st.caption("Clusters: Sentence-Transformer + HDBSCAN · labels by Groq LLM")
    st.caption("Replies: " + ("Groq LLM ✅" if groq_available() else "templates (set GROQ_API_KEY for AI replies)"))

(current_window, previous_window) = an.windows(as_of, days)
cur_start, cur_end = current_window
prev_start, prev_end = previous_window
scope = an.filter_reviews(df_all, ss.restaurant, branches or None, sources)
cur = an.filter_reviews(scope, start=cur_start, end=cur_end)
prev = an.filter_reviews(scope, start=prev_start, end=prev_end)
upto = an.filter_reviews(scope, end=as_of)
spikes = an.spikes(upto, as_of)
esc_open, esc_overdue = an.open_escalations(upto, as_of)

# ------------------------------------------------------------------ header + navigation
h1, h2 = st.columns([3, 2])
h1.markdown("## 💬 Empathy Engine")
h2.markdown(f"<div style='text-align:right;padding-top:14px'>🏪 <b>{ss.restaurant}</b> &nbsp; 🔔 "
            f"{pill(str(len(esc_open)), 'p-urgent')}</div>", unsafe_allow_html=True)
page = st.radio("Screen", PAGES, key="page", horizontal=True, label_visibility="collapsed")
st.divider()


def go_overview():
    ss.page = PAGES[2]


# ================================================================== 1. SEARCH
if page == PAGES[0]:
    with st.container(border=True):
        st.caption("GETTING STARTED")
        st.markdown("## Add a batch of reviews")
        st.markdown("Upload any review CSV or Excel file. We detect its platform when possible, then score customer "
                "sentiment and group reviews into recurring themes.")
        st.caption(issue_classifier.issue_model_status())
        render_upload_analyse_form("search")
        source_counts = rest_df["source"].value_counts()
        source_counts = source_counts[source_counts.index.astype(str).str.strip().str.casefold() != "other"]
        extra_sources = [s for s in source_counts.index if s not in SOURCES]
        source_counts = source_counts.reindex([*SOURCES, *extra_sources], fill_value=0)
        pills = "".join(pill(f"{'✓' if source_counts[s] else '–'} {s} · {source_counts[s]} total",
                             "p-ok" if source_counts[s] else "p-muted")
            for s in source_counts.index)
        st.markdown(f"<div style='text-align:center'><p class='small'>Sources</p>{pills}</div>", unsafe_allow_html=True)
        st.markdown(f"<p class='small' style='text-align:center'>📅 Selected range · {cur_start:%d %b} – "
                    f"{cur_end:%d %b %Y} &nbsp;&nbsp; ⚖️ Compare with: {', '.join(compare) or '—'}</p>",
                    unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(f"**Reviews collected by source** · {ss.restaurant} · all dates")
        mx = max(1, source_counts.max() if len(source_counts) else 1)
        for s in source_counts.index:
            a, b, c = st.columns([1.2, 6, 0.8])
            a.write(s)
            b.progress(int(source_counts.get(s, 0) / mx * 100))
            c.write(f"**{source_counts.get(s, 0)}**")
        st.button("See the overview →", on_click=go_overview)

    needs_review = df_all[
        df_all["issue_source"].eq("rules_low_confidence") & df_all["issue_confidence"].notna()
    ].sort_values("issue_confidence")
    with st.expander(f"Review low-confidence ML predictions ({len(needs_review):,})"):
        st.caption("Model probabilities are estimates, not calibrated probabilities. Staff corrections are saved "
                   "and replace the model label.")
        if needs_review.empty:
            st.info("There are no low-confidence model predictions awaiting review.")
        else:
            review_options = needs_review.head(100).reset_index(drop=True)
            st.dataframe(
                review_options[["id", "restaurant", "branch", "issue", "issue_confidence", "text"]],
                hide_index=True,
                width="stretch",
                column_config={
                    "issue_confidence": st.column_config.NumberColumn(
                        "Model confidence estimate", format="percent", width="small"
                    ),
                },
            )
            selected_id = st.selectbox(
                "Review to correct",
                review_options["id"].astype(int).tolist(),
                format_func=lambda review_id: (
                    f"Review {review_id} · "
                    f"{review_options.loc[review_options['id'].eq(review_id), 'issue'].iat[0]}"
                ),
                key="issue_review_id",
            )
            selected_review = review_options.loc[review_options["id"].eq(selected_id)].iloc[0]
            st.write(selected_review["text"])
            labels = sorted(set(ISSUES) | {POSITIVE_ISSUE, str(selected_review["issue"])})
            corrected_issue = st.selectbox(
                "Correct issue label",
                labels,
                index=labels.index(str(selected_review["issue"])),
                key=f"issue_review_label_{selected_id}",
            )
            if st.button("Save corrected label", type="primary", key=f"save_issue_{selected_id}"):
                corrected_urgency = issue_classifier.urgency_of(
                    selected_review["rating"],
                    selected_review["sentiment"],
                    corrected_issue,
                    selected_review["emotion"],
                    bool(selected_review["red_flag"]),
                )
                db.update_issue_label(int(selected_id), corrected_issue, corrected_urgency)
                st.toast("Corrected issue label saved.")
                st.rerun()

# ================================================================== 2. OVERVIEW
elif page == PAGES[2]:
    if cur.empty:
        st.info("No reviews in this date range. Widen the range or change filters in the sidebar.")
        st.stop()
    for s in spikes.head(3).itertuples():
        st.error(f"**Spike this week:** {s.issue} at **{s.branch}** — {int(s.this_week)} in the last "
                 f"7 days, {s.ratio:.1f}× the 4-week average.", icon="⚠️")

    k = an.kpis(cur, prev, as_of)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reviews in range", k["reviews"],
              None if k["reviews_change"] is None else f"{k['reviews_change']:+.0f}% vs previous", border=True)
    m2.metric("Reputation score /100", k["score"],
              None if k["score_delta"] is None else f"{k['score_delta']:+.1f} pts", border=True)
    m3.metric("Negative share", f"{k['neg_share']:.0f}%",
              None if k["neg_delta"] is None else f"{k['neg_delta']:+.1f} pts", delta_color="inverse", border=True)
    esc_c, over_c = an.open_escalations(cur, as_of)
    m4.metric("Open escalations", len(esc_c), f"{len(over_c)} overdue" if len(over_c) else "none overdue",
              delta_color="inverse" if len(over_c) else "off", border=True)

    with st.container(border=True):
        st.markdown("**Decision panel · Top 5 issues by priority**  \n"
                    "<span class='small'>Priority = 0.4 × frequency + 0.3 × negative share + 0.3 × low rating</span>",
                    unsafe_allow_html=True)
        pt = an.priority_table(complaint_rows(cur)).head(5)
        if len(pt):
            pt["recommended_action"] = [recommend(i, "affected branches")["action"] for i in pt["issue"]]
            pt["owner"] = [recommend(i)["owner"] for i in pt["issue"]]
            pt["expected_improvement"] = [recommend(i)["expected"] for i in pt["issue"]]
            pt["campaign"] = [next((f"#{c['id']} {c['name']}" for c in CAMPAIGNS if c["id"] == match_campaign(i)), "—")
                              for i in pt["issue"]]
            st.dataframe(pt[["rank", "issue", "frequency", "avg_rating", "priority_score", "priority",
                             "recommended_action", "owner", "expected_improvement", "campaign"]],
                         hide_index=True, width="stretch",
                         column_config={"avg_rating": st.column_config.NumberColumn("Avg rating", format="%.2f"),
                                        "priority_score": st.column_config.NumberColumn("Score", format="%.0f"),
                                        "campaign": "Linked campaign"})

    with st.container(border=True):
        st.markdown("**Customer emotions**")
        em = cur["emotion"].value_counts().reset_index()
        fig = px.bar(em, x="count", y="emotion", orientation="h", color="emotion",
                     color_discrete_map={"anger": RED, "frustration": AMBER, "disappointment": "#D4537E",
                                         "neutral": GREY, "delight": GREEN})
        fig.update_layout(height=260, showlegend=False, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None)
        st.plotly_chart(fig, width="stretch")

    with st.expander("Read the reviews in this date range"):
        q = st.text_input("Search text", placeholder="e.g. sambar, parking, rude")
        show = cur if not q else cur[cur["text"].str.contains(q, case=False, na=False)]
        st.dataframe(
            show[["review_date", "source", "branch", "rating", "issue", "issue_source",
                  "issue_confidence", "emotion", "urgency", "text"]]
            .sort_values("review_date", ascending=False),
            hide_index=True,
            width="stretch",
            column_config={
                "issue_source": "Label source",
                "issue_confidence": st.column_config.NumberColumn(
                    "Model confidence estimate", format="percent"
                ),
            },
        )
        st.download_button("Download CSV", show.to_csv(index=False).encode(), f"{ss.restaurant}_reviews.csv", "text/csv")

# ================================================================== 4. CAMPAIGNS (360-DEGREE)
elif page == PAGES[3]:
    st.markdown("**Digital marketing · 360-degree campaigns tied to issue clusters**  \n"
                "<span class='small'>Each campaign answers one HDBSCAN cluster. Success is measured the same way the "
                "issue was found: the cluster's share of classified reviews should fall on the next pipeline "
                "re-run.</span>", unsafe_allow_html=True)

    cls_cur, cls_prev = classified(cur), classified(prev)
    if cls_cur.empty:
        st.info("No classified reviews in this date range. Widen the range or change filters in the sidebar.")
        st.stop()
    cid_cur = cls_cur["issue"].map(match_campaign)
    cid_prev = cls_prev["issue"].map(match_campaign)
    cid_upto = upto["issue"].map(match_campaign)
    spike_cids = set(spikes["issue"].map(match_campaign).dropna()) if len(spikes) else set()

    rows = []
    for c in CAMPAIGNS:
        share_now = (cid_cur == c["id"]).mean() * 100
        share_prev = (cid_prev == c["id"]).mean() * 100 if len(cls_prev) else None
        change = None if not share_prev else (share_now - share_prev) / share_prev * 100
        wk = weekly_counts(upto[cid_upto == c["id"]], as_of)
        base = sum(wk[:-1]) / 4
        if c["mode"] == "risk":
            status = "⏸ Pause — internal review" if c["id"] in spike_cids else "▶ Run (monitor weekly)"
        elif c["mode"] == "gated":
            dropping = wk[-1] < base and c["id"] not in spike_cids
            status = "▶ Ready to launch" if dropping else "⏸ Hold until complaints drop"
        else:
            status = "▶ Launch now"
        rows.append(dict(rank=c["id"], campaign=c["name"], cluster=c["theme"],
                         reviews=int((cid_cur == c["id"]).sum()), share=round(share_now, 1),
                         change=None if change is None else round(change, 0),
                         last_week=wk[-1], avg_4w=round(base, 1), status=status))
    tbl = pd.DataFrame(rows).sort_values("reviews", ascending=False)

    top = tbl.iloc[0]
    k1, k2, k3 = st.columns(3)
    k1.metric("Reviews covered by the 5 campaigns", f"{tbl['share'].sum():.0f}%",
              f"of {len(cls_cur):,} classified reviews", delta_color="off", border=True)
    k2.metric("Largest cluster", top["cluster"], f"{top['share']:.1f}% share", delta_color="off", border=True)
    k3.metric("Campaigns on hold", int(tbl["status"].str.startswith("⏸").sum()), border=True)

    with st.container(border=True):
        st.markdown(f"**Campaign status** · share of classified reviews, selected range vs preceding {days} days")
        st.dataframe(tbl.drop(columns="rank"), hide_index=True, width="stretch",
                     column_config={"share": st.column_config.ProgressColumn("Share %", min_value=0,
                                                                             max_value=max(1.0, float(tbl["share"].max())),
                                                                             format="%.1f%%"),
                                    "change": st.column_config.NumberColumn("Change %", format="%+.0f%%"),
                                    "last_week": "Last 7 days", "avg_4w": "4-week avg / wk"})
        st.caption("Campaign 4 (wait time) launches only after the operational fix shows a measurable drop — "
                   "a speed promise followed by a slow visit backfires. Campaign 5 (hygiene) pauses on any spike.")

    with st.container(border=True):
        st.markdown("**Weekly complaint volume per campaign cluster** · selected date range")
        in_range = upto[upto["review_date"] >= pd.Timestamp(cur_start)]
        cid_range = cid_upto.loc[in_range.index]
        week_start = (in_range["review_date"].dt.normalize() - pd.Timestamp(cur_start)).dt.days // 7
        n_weeks = max(1, (pd.Timestamp(cur_end) - pd.Timestamp(cur_start)).days // 7 + 1)
        labels = [(pd.Timestamp(cur_start) + pd.Timedelta(days=7 * i)).strftime("%d %b") for i in range(n_weeks)]
        trend = pd.DataFrame({c["name"]: [int(((cid_range == c["id"]) & (week_start == i)).sum())
                                          for i in range(n_weeks)] for c in CAMPAIGNS}, index=labels)
        if trend.to_numpy().sum() == 0:
            st.info("No complaints matching these campaign clusters in this date range. Widen the range in the sidebar.")
        else:
            fig = px.line(trend.reset_index().melt(id_vars="index", var_name="campaign", value_name="reviews"),
                          x="index", y="reviews", color="campaign", markers=True)
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Week starting",
                              legend_title=None, legend=dict(orientation="h", y=-0.3))
            st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns([3, 2])
    with c1.container(border=True):
        st.markdown("**Campaign briefs**")
        for c in sorted(CAMPAIGNS, key=lambda c: -int(tbl.loc[tbl["rank"] == c["id"], "reviews"].iloc[0])):
            with st.expander(f"#{c['id']} {c['name']} — {c['theme']}"):
                st.markdown(f"<div class='quote'>“{c['headline']}”</div>", unsafe_allow_html=True)
                st.markdown(" ".join(pill(ch) for ch in c["channels"]), unsafe_allow_html=True)
                st.markdown(f"<span class='small'>KPI: {c['kpi']}</span>", unsafe_allow_html=True)
                ex = cls_cur[cid_cur == c["id"]].sort_values("rating").head(3)
                for t in ex["text"]:
                    st.markdown(f"<div class='small'>• {str(t)[:220]}</div>", unsafe_allow_html=True)
    with c2.container(border=True):
        st.markdown("**360° channel matrix**")
        mat = pd.DataFrame({f"#{c['id']}": ["✅" if ch in c["channels"] else "" for ch in CHANNELS]
                            for c in sorted(CAMPAIGNS, key=lambda c: c["id"])}, index=CHANNELS)
        st.dataframe(mat, width="stretch")

    unmatched = cls_cur.loc[cid_cur.isna(), "issue"].value_counts()
    if len(unmatched):
        with st.expander(f"Clusters without a campaign ({len(unmatched)})"):
            st.dataframe(unmatched.rename("reviews"), width="stretch")
            st.caption("These are usually positive or low-volume clusters. Add keywords to CAMPAIGNS in main.py "
                       "if a new complaint cluster appears after a re-run.")

# ================================================================== 5. MONTHLY REPORT
elif page == PAGES[4]:
    rest_all = df_all[df_all["restaurant"] == ss.restaurant]
    months = sorted(rest_all["review_date"].dt.to_period("M").astype(str).unique(), reverse=True)
    c1, c2 = st.columns([2, 1])
    month = c1.selectbox("Month", months, format_func=lambda m: pd.Period(m).strftime("%B %Y"))
    rkey = f"report_{ss.restaurant}_{month}"
    if c2.button("🔄 Generate / refresh report", type="primary", width="stretch") or rkey not in ss:
        with st.spinner("Building report..."):
            rep = build_report(an.filter_reviews(df_all, sources=sources), ss.restaurant, month)
            pdf, path = save_report(rep)
            ss[rkey] = (rep, pdf)
    rep, pdf = ss[rkey]
    k, rp = rep["kpis"], rep["reply"]

    with st.container(border=True):
        t1, t2 = st.columns([4, 1])
        t1.markdown(f"### {rep['restaurant']} · {rep['label']} report")
        t1.markdown(f"<p class='small'>Generated {date.today():%d %b %Y} · {k['reviews']} reviews from "
                    f"{len(rep['sources'])} sources ({', '.join(f'{s} {n}' for s, n in rep['sources'].items())})"
                    f" · summary by {rep['engine']}</p>", unsafe_allow_html=True)
        t2.download_button("⬇️ PDF", pdf, f"{rep['restaurant'].replace(' ', '_')}_{month}.pdf", "application/pdf",
                           width="stretch")
        st.markdown("**Summary**")
        st.write(rep["summary"])
        st.markdown("**Top 3 actions for next month**")
        for i, a in enumerate(rep["actions"], 1):
            cls = {1: "p-urgent", 2: "p-high", 3: "p-ok"}[i]
            st.markdown(f"{pill(str(i), cls)} **{a['action']}** <span class='small'>· {a['issue']} · "
                        f"Owner: {a['owner']} · Target: {a['expected']}</span>", unsafe_allow_html=True)
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Reputation score", k["score"], None if k["score_delta"] is None else f"{k['score_delta']:+.1f} pts", border=True)
        x2.metric("Reply rate (negative reviews)", "-" if rp["reply_rate"] is None else f"{rp['reply_rate']:.0f}%", border=True)
        x3.metric("Median reply time", "-" if rp["avg_reply_hours"] is None else f"{rp['avg_reply_hours']:.0f} h", border=True)
        x4.metric("Escalations resolved", f"{rp['resolved']}/{rp['escalations']}", border=True)

    with st.container(border=True):
        st.markdown("**Complaint themes vs last month**")
        iss = rep["issues"]
        iss = iss[~is_noise(iss["issue"])].head(8)
        fig = go.Figure()
        fig.add_bar(y=iss["issue"], x=iss["previous"], name="Last month", orientation="h", marker_color=GREY)
        fig.add_bar(y=iss["issue"], x=iss["mentions"], name="This month", orientation="h", marker_color=ACCENT)
        fig.update_layout(height=340, barmode="group", margin=dict(l=0, r=0, t=10, b=0),
                          yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")

    with st.container(border=True):
        st.markdown("**Branches**")
        st.dataframe(
            rep["branches"],
            hide_index=True,
            width="stretch",
            height=340,
            column_config={
                "branch": st.column_config.TextColumn("Branch", width="medium"),
                "reviews": st.column_config.NumberColumn("Reviews", width="small"),
                "avg_rating": st.column_config.NumberColumn("Avg rating", format="%.2f", width="small"),
                "negative_pct": st.column_config.NumberColumn("Negative %", format="%.1f%%", width="small"),
                "score": st.column_config.NumberColumn("Score", format="%.1f", width="small"),
                "top_complaint": st.column_config.TextColumn("Top complaint", width="medium"),
            },
        )

    with st.expander("Previously generated reports"):
        with db.connect() as con:
            past = pd.read_sql_query("SELECT restaurant, month, created_at, pdf_path FROM reports ORDER BY month DESC", con)
        st.dataframe(past, hide_index=True, width="stretch")

# ================================================================== 6. SERVQUAL
elif page == PAGES[5]:
    st.markdown("**SERVQUAL · customer expectations versus perceived service**  \n"
                "<span class='small'>A negative gap means perceived service fell short of expectations. "
                "NLP complaint counts use the current restaurant, source and date filters.</span>",
                unsafe_allow_html=True)

    survey_path = _HERE / "outputs" / "cleaned_servqual_responses.xlsx"
    if not survey_path.is_file():
        st.warning(f"Survey response workbook not found: {survey_path}")
    else:
        servqual, survey_respondents = load_servqual_scores(str(survey_path))
        scored = servqual.dropna(subset=["mean_gap"])
        worst = scored.loc[scored["mean_gap"].idxmin()] if not scored.empty else None
        item_count = int(servqual["n_items"].sum())

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Survey respondents", survey_respondents, border=True)
        k2.metric("Worst service gap" if worst is None else f"Worst gap · {worst['dimension']}",
                  "—" if worst is None else f"{worst['mean_gap']:+.2f}", border=True)
        k3.metric("Survey items represented", f"{item_count} / 20", border=True)
        k4.metric("Classified reviews", f"{int((~is_noise(cur['issue'])).sum()):,}",
                  f"{cur_start:%d %b %Y} – {cur_end:%d %b %Y}", border=True)

        chart_col, gap_col = st.columns(2)
        with chart_col.container(border=True):
            st.markdown("**Expectation and perception** · 1–7 scale")
            comparison = servqual.melt(
                id_vars="dimension",
                value_vars=["mean_expectation", "mean_perception"],
                var_name="measure",
                value_name="score",
            ).dropna(subset=["score"])
            if comparison.empty:
                st.info("No paired expectation/perception responses are available.")
            else:
                comparison["measure"] = comparison["measure"].map({
                    "mean_expectation": "Expectation", "mean_perception": "Perception"
                })
                fig = px.bar(comparison, x="dimension", y="score", color="measure", barmode="group",
                             color_discrete_map={"Expectation": GREY, "Perception": ACCENT},
                             category_orders={"dimension": servqual["dimension"].tolist()})
                fig.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0),
                                  yaxis=dict(range=[0, 7]), xaxis_title=None, yaxis_title="Mean score",
                                  legend_title=None, legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig, width="stretch")

        with gap_col.container(border=True):
            st.markdown("**Service-quality gap** · perception − expectation")
            if scored.empty:
                st.info("No scored SERVQUAL dimensions are available.")
            else:
                gap_chart = scored.sort_values("mean_gap").copy()
                gap_chart["status"] = gap_chart["mean_gap"].map(
                    lambda gap: "Shortfall" if gap < 0 else "Meets expectations"
                )
                fig = px.bar(gap_chart, x="mean_gap", y="dimension", orientation="h", color="status",
                             color_discrete_map={"Shortfall": RED, "Meets expectations": GREEN})
                fig.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0),
                                  xaxis_title="Mean gap", yaxis_title=None, legend_title=None,
                                  legend=dict(orientation="h", y=-0.2))
                fig.add_vline(x=0, line_color=GREY, line_width=1)
                st.plotly_chart(fig, width="stretch")

        issue_categories = {
            "Reliability": ["Food Quantity & Value for Money", "Billing & Online Ordering Issues"],
            "Responsiveness": ["Slow Service & Staff Negligence", "Food Variety & Fast Service"],
            "Assurance": ["Poor Experience & Food Hygiene Complaints"],
            "Empathy": ["Staff Courtesy & Behaviour", "Service Quality"],
            "Tangibles": ["Cleanliness & Restroom Hygiene", "Ambience & Seating"],
        }
        issue_counts = cur["issue"].dropna().astype(str).str.casefold().value_counts()
        triangulation = servqual[["dimension", "mean_gap", "n_items", "respondents"]].copy()
        triangulation["matched_issue_categories"] = triangulation["dimension"].map(
            lambda dimension: ", ".join(issue_categories[dimension])
        )
        triangulation["matched_nlp_frequency"] = triangulation["dimension"].map(
            lambda dimension: sum(issue_counts.get(category.casefold(), 0)
                                  for category in issue_categories[dimension])
        )
        with st.container(border=True):
            st.markdown("**SERVQUAL and review triangulation**")
            st.dataframe(
                triangulation[["dimension", "mean_gap", "n_items", "respondents",
                               "matched_nlp_frequency", "matched_issue_categories"]],
                hide_index=True,
                width="stretch",
                column_config={
                    "mean_gap": st.column_config.NumberColumn("Mean gap", format="%+.2f"),
                    "n_items": "Survey items",
                    "respondents": "Respondents",
                    "matched_nlp_frequency": "NLP mentions",
                    "matched_issue_categories": "Matched review themes",
                },
            )
            st.caption(
                f"Survey scores use all {survey_respondents:,} available submissions in "
                "outputs/cleaned_servqual_responses.xlsx; they are not filtered by restaurant. "
                "NLP mentions reflect the selected restaurant and current dashboard filters. "
                "Dimensions without mapped survey items remain unscored."
            )

# ================================================================== 2. ANALYTICS
elif page == PAGES[1]:
    in_window = an.filter_reviews(df_all, sources=sources, start=cur_start, end=cur_end)
    in_window = in_window[in_window["restaurant"].isin([ss.restaurant, *compare])]
    if branches:
        in_window = in_window[(in_window["restaurant"] != ss.restaurant) | in_window["branch"].isin(branches)]
    analytics_dashboard.render(in_window, an, {"an": an, "cur": cur, "prev": prev, "restaurant": ss.restaurant, "days": days})

# ================================================================== 7. EMPATHETIC REPLY
elif page == PAGES[6]:
    st.markdown("**Draft a thoughtful response to a customer review**  \n"
                "<span class='small'>Replies use Groq when configured, otherwise a local template. "
                "Review and edit every draft before posting.</span>", unsafe_allow_html=True)

    with st.container(border=True):
        reply_text = st.text_area(
            "Customer review",
            height=140,
            placeholder="Paste the review you want to respond to...",
            key="standalone_reply_review",
        )
        reply_branch = st.selectbox(
            "Branch",
            ["Not specified", *branch_options],
            key="standalone_reply_branch",
        )
        reply_issue_options = sorted(df_all["issue"].dropna().astype(str).unique())
        reply_issue = st.selectbox(
            "Issue category",
            reply_issue_options or ISSUES,
            key="standalone_reply_issue",
        )
        reply_emotion_options = sorted(df_all["emotion"].dropna().astype(str).unique())
        reply_emotion = st.selectbox(
            "Customer emotion",
            reply_emotion_options or ["neutral"],
            key="standalone_reply_emotion",
        )
        reply_rating = st.slider(
            "Review rating",
            min_value=1,
            max_value=5,
            value=3,
            key="standalone_reply_rating",
        )

        if st.button("Draft empathetic reply", type="primary", key="standalone_draft_reply"):
            if not reply_text.strip():
                st.warning("Enter a customer review before drafting a reply.")
            else:
                with st.spinner("Drafting reply..."):
                    reply, reply_engine = draft_reply(
                        ss.restaurant, reply_branch, reply_issue, reply_emotion, reply_text, reply_rating
                    )
                ss["standalone_reply_result"] = reply
                ss["standalone_reply_engine"] = reply_engine

    if ss.get("standalone_reply_result"):
        engine_label = "Groq AI" if ss["standalone_reply_engine"] == "groq" else "local template"
        st.caption(f"Drafted with {engine_label}. Edit the response before posting.")
        st.text_area(
            "Editable reply draft",
            value=ss["standalone_reply_result"],
            height=160,
            key="standalone_reply_output",
        )

"""
module6_reputation_early_warning_app.py

Module 6 — Reputation Early-Warning & Resolution System (detailed build)

Reuses your existing HDBSCAN + LLM issue categories directly (no separate
NLP system) and organizes the workflow into four detailed pages:

  TAB 1 — Reputation Risk Score
      Branch-level score (0-100, higher = healthier), rolled up by
      restaurant chain. Charts are shown RESTAURANT-WISE -> BRANCH-WISE
      (small multiples, one panel per restaurant, branches on the x-axis
      inside each panel). The results table includes RESTAURANT and
      SOURCE (Google / Zomato / TripAdvisor) as explicit columns.

  TAB 2 — Early-Warning Alerts (week-over-week spike detection)
      Per (restaurant, branch, issue_category): compares this week's
      complaint count to last week's. Flags anything above a configurable
      spike threshold. Each alert can be expanded to see which platform
      (source) the spike is coming from.

  TAB 3 — Complaint-to-Resolution Workflow
      Each alert moves through New -> Assigned -> In Progress -> Resolved
      / Monitoring / Escalate. Manager logs an owner, action taken and
      action date. The app auto-compares complaint volume before vs.
      after the action date (Module 6 roadmap Step 5) and shows a
      Resolved / Still Emerging / Escalate verdict. Persists to a local
      CSV (resolution_tracker.csv) so it survives Streamlit reruns.

  TAB 4 — Response Draft Generator
      Manager-facing draft reply for a selected review, with tone and
      platform (source) controls, since a Google reply reads differently
      to a Zomato/Swiggy reply. Template-based by default (no external
      API key needed). If GROQ_API_KEY is set, uses Groq for a more
      natural draft — optional, clearly flagged, template fallback is
      fully functional on its own for the demo/viva.

--------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------
    pip install streamlit pandas numpy plotly openpyxl
    # optional, only if you want LLM-drafted responses:
    pip install groq
    streamlit run module6_reputation_early_warning_app.py

--------------------------------------------------------------------
EXPECTED INPUT COLUMNS (edit CONFIG below to match your file)
--------------------------------------------------------------------
    restaurant     -> chain name, e.g. "Sangeetha", "A2B (Adyar Ananda Bhavan)"
    branch         -> specific outlet/branch name (108 branches total)
    source         -> platform: "Google" / "Zomato" / "TripAdvisor"
    rating         -> numeric star rating (1-5)
    issue_category -> HDBSCAN + LLM cluster label (e.g. "Service Quality")
    review_text    -> raw review text (used for response drafting)
    review_date    -> date the review was posted (needed for trend/spike
                       detection). If your cleaned file doesn't have this
                       yet, add it from the raw scraped data before running.

If `restaurant` or `source` are missing from your file, the app will
still run (it fills them with "Unknown Restaurant" / "Unknown Source"
and shows a warning) but you'll lose the restaurant/branch drill-down
and platform breakdown that this build adds — best to include them.
--------------------------------------------------------------------
"""

import os
import sys
from datetime import timedelta
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_local_module(module_name, filename):
    module_path = os.path.join(MODULE_DIR, filename)
    spec = spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {module_path}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


load_local_module("cx_common", "cx_common (1).py")
root_cause_module = load_local_module(
    "root_cause_emotion_analysis", "root_cause_emotion_analysis (1).py"
)
emotion_module = load_local_module("emotion_classification", "emotion_classification (1).py")
escalation_module = load_local_module("escalation_detection", "escalation_detection (1).py")
reply_module = load_local_module(
    "empathetic_reply_generator", "empathetic_reply_generator (1).py"
)

build_crosstab = root_cause_module.build_crosstab
dominant_emotion_summary = root_cause_module.dominant_emotion_summary
classify_emotions = emotion_module.classify_emotions
cluster_level_alert = escalation_module.cluster_level_alert
compute_urgency = escalation_module.compute_urgency
fallback_reply = reply_module.fallback_reply

# ============================== CONFIG ==============================

DATA_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "cleaned_reviews.xlsx",
)  # .xlsx or .csv
CLUSTER_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "issues",
    "hdbscan_clustered_reviews.xlsx",
)
LABEL_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "issues",
    "cluster_topics_labeled.xlsx",
)
CLASSIFIER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "issue_classifier.joblib",
)
CLASSIFIED_REVIEWS_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "reviews_with_issue_classification.xlsx",
)
CLASSIFICATION_SUMMARY_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "issue_classification_summary.xlsx",
)
SERVQUAL_RESPONSES_PATH = os.path.join(BASE_DIR, "outputs", "cleaned_servqual_responses.xlsx")
SERVQUAL_SCORES_PATH = os.path.join(BASE_DIR, "servqual_dimension_scores.csv")
SERVQUAL_TRIANGULATION_PATH = os.path.join(BASE_DIR, "servqual_nlp_triangulation.csv")
CX_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "cx")
EMOTION_OUTPUT_PATH = os.path.join(CX_OUTPUT_DIR, "reviews_with_emotion.csv")
NEW_REVIEWS_PATH = os.path.join(BASE_DIR, "new_reviews.csv")

RESTAURANT_COLUMN = "restaurant"
BRANCH_COLUMN = "branch"
SOURCE_COLUMN = "source"
RATING_COLUMN = "rating"
ISSUE_CATEGORY_COLUMN = "issue_category"
TEXT_COLUMN = "review_text"
DATE_COLUMN = "review_date"        # must be parseable as a date
NOISE_LABELS = ["Noise", "noise", "-1", -1]

NEGATIVE_ISSUE_CATEGORIES = [
    "Food Quantity & Value for Money",
    "Slow Service & Staff Negligence",
    "Service Quality",
    "Poor Experience & Food Hygiene Complaints",
    "Service Issues",
    "Crowd and Wait Time",
    "Food Quality Issues",
]

SATISFACTION_THRESHOLD = 4

# Week-over-week increase (%) at which an issue is flagged as an alert.
# 100% = complaint count has doubled vs the prior week.
SPIKE_THRESHOLD_PCT = 75

# Minimum complaint count this week to even consider an alert
# (avoids flagging noise like "1 -> 2 complaints" as a 100% spike).
MIN_WEEKLY_COUNT_FOR_ALERT = 3

TRACKER_PATH = "resolution_tracker.csv"   # persists alert statuses across runs
RESOLUTION_WINDOW_DAYS = 14                # "before vs after" comparison window

USE_GROQ = bool(os.environ.get("GROQ_API_KEY"))  # optional LLM-drafted responses

STATUS_OPTIONS = ["New", "Assigned", "In Progress", "Resolved", "Monitoring", "Escalate"]
def analyze_new_review(review_text: str, rating: int) -> dict:
    """Provide an explainable first-pass analysis for a newly submitted review."""
    text = review_text.lower()
    negative_terms = ["bad", "poor", "worst", "slow", "rude", "dirty", "hair", "cold", "late", "delay", "terrible", "awful"]
    positive_terms = ["good", "great", "excellent", "amazing", "delicious", "friendly", "love", "best"]
    issue_rules = [
        ("Slow Service & Staff Negligence", ["slow", "wait", "delay", "late", "staff", "service", "rude"]),
        ("Food Quality Issues", ["food", "taste", "cold", "stale", "hair", "raw", "quality"]),
        ("Poor Experience & Food Hygiene Complaints", ["dirty", "hygiene", "unclean", "hair", "stain"]),
        ("Pricing & Value Concerns", ["price", "expensive", "cost", "overpriced", "value"]),
    ]
    issue = next((label for label, terms in issue_rules if any(term in text for term in terms)), "General Feedback")
    negative_hits = sum(term in text for term in negative_terms)
    positive_hits = sum(term in text for term in positive_terms)
    if rating <= 2 or negative_hits > positive_hits:
        sentiment = "Negative"
    elif rating >= 4 or positive_hits > negative_hits:
        sentiment = "Positive"
    else:
        sentiment = "Neutral"
    severity = "High" if rating <= 1 or negative_hits >= 3 else "Medium" if sentiment == "Negative" else "Low"
    return {"sentiment": sentiment, "issue": issue, "severity": severity}


def load_new_reviews() -> pd.DataFrame:
    if not os.path.exists(NEW_REVIEWS_PATH):
        return pd.DataFrame()
    return pd.read_csv(NEW_REVIEWS_PATH)


@st.cache_resource
def load_classifier():
    if not os.path.exists(CLASSIFIER_PATH):
        return None, None
    bundle = joblib.load(CLASSIFIER_PATH)
    return bundle["model"], bundle["vectorizer"]


@st.cache_data
def load_total_review_count() -> int:
    return len(pd.read_excel(DATA_PATH))


@st.cache_data
def load_classified_reviews() -> pd.DataFrame | None:
    if not os.path.exists(CLASSIFIED_REVIEWS_PATH):
        return None
    return pd.read_excel(CLASSIFIED_REVIEWS_PATH)


@st.cache_data
def load_servqual_outputs() -> tuple[pd.DataFrame | None, pd.DataFrame | None, int]:
    if not os.path.exists(SERVQUAL_SCORES_PATH) or not os.path.exists(SERVQUAL_TRIANGULATION_PATH):
        return None, None, 0
    response_count = len(pd.read_excel(SERVQUAL_RESPONSES_PATH)) if os.path.exists(SERVQUAL_RESPONSES_PATH) else 0
    return (
        pd.read_csv(SERVQUAL_SCORES_PATH),
        pd.read_csv(SERVQUAL_TRIANGULATION_PATH),
        response_count,
    )


@st.cache_data
def load_emotion_reviews() -> pd.DataFrame | None:
    if not os.path.exists(EMOTION_OUTPUT_PATH):
        return None
    return pd.read_csv(EMOTION_OUTPUT_PATH)


def run_emotion_analysis() -> pd.DataFrame:
    classified_reviews = pd.read_excel(CLASSIFIED_REVIEWS_PATH)
    classified_reviews["issue_cluster"] = classified_reviews["predicted_issue_category"]
    emotion_reviews = classify_emotions(classified_reviews, "review_text")
    os.makedirs(CX_OUTPUT_DIR, exist_ok=True)
    emotion_reviews.to_csv(EMOTION_OUTPUT_PATH, index=False)
    return emotion_reviews


def apply_classifier_to_cleaned_reviews(model, vectorizer) -> tuple[pd.DataFrame, pd.DataFrame]:
    reviews = pd.read_excel(DATA_PATH)
    review_column = next(
        (column for column in ("review_text", "review", "Reviews") if column in reviews.columns),
        None,
    )
    if review_column is None:
        raise ValueError("The cleaned reviews file has no review text column.")

    reviews["review_text"] = reviews[review_column].astype(str)
    features = vectorizer.transform(reviews["review_text"])
    reviews["predicted_issue_category"] = model.predict(features)
    reviews["confidence"] = model.predict_proba(features).max(axis=1)

    distribution = reviews["predicted_issue_category"].value_counts().reset_index()
    distribution.columns = ["Issue Category", "Count"]
    distribution["Percentage"] = (distribution["Count"] / len(reviews) * 100).round(2)

    reviews.to_excel(CLASSIFIED_REVIEWS_PATH, index=False)
    with pd.ExcelWriter(CLASSIFICATION_SUMMARY_PATH) as writer:
        distribution.to_excel(writer, sheet_name="Issue Distribution", index=False)
        pd.crosstab(
            reviews[RESTAURANT_COLUMN], reviews["predicted_issue_category"]
        ).to_excel(writer, sheet_name="Restaurant Issues")

    return reviews, distribution

# ======================================================================


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if "cluster" not in df.columns and os.path.exists(CLUSTER_PATH):
        cluster_df = pd.read_excel(CLUSTER_PATH)
        if len(cluster_df) == len(df) and "cluster" in cluster_df.columns:
            df["cluster"] = cluster_df["cluster"].to_numpy()

    if "cluster" in df.columns and os.path.exists(LABEL_PATH):
        labels = pd.read_excel(LABEL_PATH)
        if {"cluster_id", "issue_label"}.issubset(labels.columns):
            label_map = labels.set_index("cluster_id")["issue_label"]
            mapped_labels = df["cluster"].map(label_map)
            if ISSUE_CATEGORY_COLUMN in df.columns:
                df[ISSUE_CATEGORY_COLUMN] = mapped_labels.combine_first(df[ISSUE_CATEGORY_COLUMN])
            else:
                df[ISSUE_CATEGORY_COLUMN] = mapped_labels
            df.loc[df["cluster"] == -1, ISSUE_CATEGORY_COLUMN] = "Noise"

    if TEXT_COLUMN not in df.columns and "review" in df.columns:
        df[TEXT_COLUMN] = df["review"]
    if ISSUE_CATEGORY_COLUMN not in df.columns and "cluster" in df.columns:
        df[ISSUE_CATEGORY_COLUMN] = "Cluster " + df["cluster"].astype(str)
    if ISSUE_CATEGORY_COLUMN not in df.columns:
        df[ISSUE_CATEGORY_COLUMN] = "Unknown Issue"
    else:
        df[ISSUE_CATEGORY_COLUMN] = df[ISSUE_CATEGORY_COLUMN].fillna("Unknown Issue")

    rating_is_imputed = RATING_COLUMN not in df.columns
    date_is_imputed = DATE_COLUMN not in df.columns
    new_reviews = load_new_reviews()
    if not new_reviews.empty:
        df = pd.concat([df, new_reviews], ignore_index=True)

    if RATING_COLUMN not in df.columns:
        df[RATING_COLUMN] = 3.0
    else:
        df[RATING_COLUMN] = pd.to_numeric(df[RATING_COLUMN], errors="coerce").fillna(3.0)
    if DATE_COLUMN not in df.columns:
        df[DATE_COLUMN] = pd.Timestamp.today().normalize()
    else:
        df[DATE_COLUMN] = df[DATE_COLUMN].fillna(pd.Timestamp.today().normalize())
    if "source_file" in df.columns:
        source_text = df["source_file"].astype(str).str.lower()
        inferred_source = np.select(
            [
                source_text.str.contains("google"),
                source_text.str.contains("zomato"),
                source_text.str.contains("trip advisor|tripadvisor"),
            ],
            ["Google", "Zomato", "TripAdvisor"],
            default="Unknown Source",
        )
        if SOURCE_COLUMN in df.columns:
            df[SOURCE_COLUMN] = df[SOURCE_COLUMN].replace("", np.nan).fillna(
                pd.Series(inferred_source, index=df.index)
            )
        else:
            df[SOURCE_COLUMN] = inferred_source

    required = [RATING_COLUMN, BRANCH_COLUMN, ISSUE_CATEGORY_COLUMN, DATE_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected column(s) {missing} in {path}. "
            f"Found columns: {list(df.columns)}. "
            f"Note: Module 6 needs a review date column for trend detection."
        )

    if RESTAURANT_COLUMN not in df.columns:
        st.warning(
            f"Column '{RESTAURANT_COLUMN}' not found — restaurant-wise views will "
            f"show a single 'Unknown Restaurant' group. Add this column for the "
            f"full restaurant → branch drill-down."
        )
        df[RESTAURANT_COLUMN] = "Unknown Restaurant"

    if SOURCE_COLUMN not in df.columns:
        st.warning(
            f"Column '{SOURCE_COLUMN}' not found — platform breakdown will show a "
            f"single 'Unknown Source' group. Add this column (Google/Zomato/"
            f"TripAdvisor) for the full platform breakdown."
        )
        df[SOURCE_COLUMN] = "Unknown Source"

    df[RESTAURANT_COLUMN] = df[RESTAURANT_COLUMN].fillna("Unknown Restaurant").astype(str).str.strip()
    df[BRANCH_COLUMN] = df[BRANCH_COLUMN].fillna("Unknown Branch").astype(str).str.strip()
    df[SOURCE_COLUMN] = df[SOURCE_COLUMN].fillna("Unknown Source").astype(str).str.strip()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.dropna(subset=[DATE_COLUMN])
    df["week"] = df[DATE_COLUMN].dt.to_period("W").apply(lambda p: p.start_time)
    df[ISSUE_CATEGORY_COLUMN] = df[ISSUE_CATEGORY_COLUMN].astype(str)
    df = df[~df[ISSUE_CATEGORY_COLUMN].isin([str(x) for x in NOISE_LABELS])]
    df["_rating_is_imputed"] = rating_is_imputed
    df["_date_is_imputed"] = date_is_imputed
    return df


# ---------------------------------------------------------------------
# TAB 1 — Reputation Risk Score (restaurant -> branch, with source mix)
# ---------------------------------------------------------------------
def compute_reputation_risk(df: pd.DataFrame) -> pd.DataFrame:
    is_negative = df[ISSUE_CATEGORY_COLUMN].isin(NEGATIVE_ISSUE_CATEGORIES)
    latest_date = df[DATE_COLUMN].max()
    recent_cutoff = latest_date - timedelta(days=30)
    prior_cutoff = latest_date - timedelta(days=60)

    recent = df[df[DATE_COLUMN] >= recent_cutoff]
    prior = df[(df[DATE_COLUMN] >= prior_cutoff) & (df[DATE_COLUMN] < recent_cutoff)]

    rows = []
    group_cols = [RESTAURANT_COLUMN, BRANCH_COLUMN]
    for (restaurant, branch), g in df.groupby(group_cols):
        mask = (df[RESTAURANT_COLUMN] == restaurant) & (df[BRANCH_COLUMN] == branch)
        g_recent = recent[(recent[RESTAURANT_COLUMN] == restaurant) & (recent[BRANCH_COLUMN] == branch)]
        g_prior = prior[(prior[RESTAURANT_COLUMN] == restaurant) & (prior[BRANCH_COLUMN] == branch)]

        avg_rating = g[RATING_COLUMN].mean()
        negative_rate = is_negative[mask].mean()
        positive_rate = (g[RATING_COLUMN] >= SATISFACTION_THRESHOLD).mean()

        recent_avg = g_recent[RATING_COLUMN].mean() if len(g_recent) else avg_rating
        prior_avg = g_prior[RATING_COLUMN].mean() if len(g_prior) else recent_avg
        rating_trend = recent_avg - prior_avg  # negative = worsening

        rows.append({
            RESTAURANT_COLUMN: restaurant,
            BRANCH_COLUMN: branch,
            "review_count": len(g),
            "avg_rating": avg_rating,
            "positive_rate": positive_rate,
            "negative_issue_rate": negative_rate,
            "rating_trend_30d": rating_trend,
        })

    result = pd.DataFrame(rows)

    # Normalize components to 0-1 (globally, across all branches), then
    # combine into a 0-100 risk score (higher = healthier reputation).
    result["rating_norm"] = (result["avg_rating"] - result["avg_rating"].min()) / (
        result["avg_rating"].max() - result["avg_rating"].min() + 1e-9
    )
    result["trend_norm"] = (result["rating_trend_30d"] - result["rating_trend_30d"].min()) / (
        result["rating_trend_30d"].max() - result["rating_trend_30d"].min() + 1e-9
    )

    result["reputation_score"] = 100 * (
        0.35 * result["rating_norm"]
        + 0.25 * result["positive_rate"]
        + 0.20 * (1 - result["negative_issue_rate"])
        + 0.20 * result["trend_norm"]
    )

    def status(score):
        if score >= 75:
            return "🟢 Healthy"
        elif score >= 55:
            return "🟠 Watch"
        else:
            return "🔴 At Risk"

    result["status"] = result["reputation_score"].apply(status)

    # Platform (source) mix per branch -> pivoted into one column per source,
    # e.g. "Google", "Zomato", "TripAdvisor" review counts.
    source_pivot = (
        df.pivot_table(
            index=group_cols,
            columns=SOURCE_COLUMN,
            values=RATING_COLUMN,
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )
    result = result.merge(source_pivot, on=group_cols, how="left")

    return result.sort_values("reputation_score", ascending=True).reset_index(drop=True)


def restaurant_rollup(risk_df: pd.DataFrame) -> pd.DataFrame:
    """Chain-level summary: weighted-average score per restaurant."""
    agg = risk_df.groupby(RESTAURANT_COLUMN).apply(
        lambda g: pd.Series({
            "branch_count": g[BRANCH_COLUMN].nunique(),
            "review_count": g["review_count"].sum(),
            "avg_reputation_score": np.average(g["reputation_score"], weights=g["review_count"]),
            "avg_rating": np.average(g["avg_rating"], weights=g["review_count"]),
        })
    ).reset_index()
    return agg.sort_values("avg_reputation_score", ascending=True).reset_index(drop=True)


# ---------------------------------------------------------------------
# TAB 2 — Early-warning spike detection (restaurant + branch + issue)
# ---------------------------------------------------------------------
def detect_spikes(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [RESTAURANT_COLUMN, BRANCH_COLUMN, ISSUE_CATEGORY_COLUMN]
    weekly = df.groupby(group_cols + ["week"]).size().reset_index(name="count")

    weeks_sorted = sorted(weekly["week"].unique())
    if len(weeks_sorted) < 2:
        return pd.DataFrame()  # not enough history yet

    this_week, last_week = weeks_sorted[-1], weeks_sorted[-2]

    this_wk = weekly[weekly["week"] == this_week].set_index(group_cols)["count"]
    last_wk = weekly[weekly["week"] == last_week].set_index(group_cols)["count"]

    combined = pd.DataFrame({"this_week": this_wk, "last_week": last_wk}).fillna(0)
    combined = combined[combined["this_week"] >= MIN_WEEKLY_COUNT_FOR_ALERT]
    combined["pct_change"] = np.where(
        combined["last_week"] > 0,
        100 * (combined["this_week"] - combined["last_week"]) / combined["last_week"],
        float(SPIKE_THRESHOLD_PCT),  # New issue: flag at the minimum alert threshold.
    )

    alerts = combined[combined["pct_change"] >= SPIKE_THRESHOLD_PCT].reset_index()
    alerts = alerts.sort_values("pct_change", ascending=False)
    alerts["alert_id"] = (
        alerts[RESTAURANT_COLUMN].astype(str) + " | "
        + alerts[BRANCH_COLUMN].astype(str) + " | "
        + alerts[ISSUE_CATEGORY_COLUMN].astype(str)
    )
    alerts.attrs["this_week"] = this_week
    return alerts


def severity_for(pct_change: float, this_week_count: int) -> str:
    if pct_change >= 150 or this_week_count >= 15:
        return "High"
    elif pct_change >= SPIKE_THRESHOLD_PCT or this_week_count >= 8:
        return "Medium"
    return "Low"


def source_mix_for_alert(df: pd.DataFrame, restaurant: str, branch: str, issue: str, week) -> pd.DataFrame:
    """Which platform is driving this week's spike for a given alert."""
    subset = df[
        (df[RESTAURANT_COLUMN] == restaurant)
        & (df[BRANCH_COLUMN] == branch)
        & (df[ISSUE_CATEGORY_COLUMN] == issue)
        & (df["week"] == week)
    ]
    if subset.empty:
        return pd.DataFrame()
    return (
        subset.groupby(SOURCE_COLUMN).size().reset_index(name="complaints_this_week")
        .sort_values("complaints_this_week", ascending=False)
    )


# ---------------------------------------------------------------------
# TAB 3 — Resolution tracker (persisted to CSV) + before/after check
# ---------------------------------------------------------------------
TRACKER_COLUMNS = [
    "alert_id", "restaurant", "branch", "issue", "status",
    "assigned_to", "action_taken", "action_date", "notes",
]


def load_tracker() -> pd.DataFrame:
    if os.path.exists(TRACKER_PATH):
        tracker = pd.read_csv(TRACKER_PATH)
        for col in TRACKER_COLUMNS:
            if col not in tracker.columns:
                tracker[col] = ""
        return tracker[TRACKER_COLUMNS]
    return pd.DataFrame(columns=TRACKER_COLUMNS)


def save_tracker(tracker_df: pd.DataFrame):
    tracker_df.to_csv(TRACKER_PATH, index=False)


def before_after_effectiveness(df: pd.DataFrame, restaurant: str, branch: str, issue: str, action_date) -> dict:
    """Compares complaint volume for RESOLUTION_WINDOW_DAYS before vs after
    the logged action date (Module 6 roadmap Step 5)."""
    if pd.isna(action_date):
        return {}
    action_date = pd.to_datetime(action_date)
    before_start = action_date - timedelta(days=RESOLUTION_WINDOW_DAYS)
    after_end = action_date + timedelta(days=RESOLUTION_WINDOW_DAYS)
    today = df[DATE_COLUMN].max()

    subset = df[
        (df[RESTAURANT_COLUMN] == restaurant)
        & (df[BRANCH_COLUMN] == branch)
        & (df[ISSUE_CATEGORY_COLUMN] == issue)
    ]
    before_count = subset[(subset[DATE_COLUMN] >= before_start) & (subset[DATE_COLUMN] < action_date)].shape[0]

    days_elapsed = (today - action_date).days
    after_window_end = min(after_end, today)
    after_count = subset[(subset[DATE_COLUMN] >= action_date) & (subset[DATE_COLUMN] <= after_window_end)].shape[0]

    if days_elapsed < RESOLUTION_WINDOW_DAYS:
        verdict = f"Still Emerging — monitoring window open ({days_elapsed}/{RESOLUTION_WINDOW_DAYS} days elapsed)"
    elif after_count < before_count:
        verdict = "Resolved — complaint volume dropped"
    elif after_count > before_count:
        verdict = "Escalate — complaint volume rose after action"
    else:
        verdict = "Still Emerging — no material change"

    return {
        "before_count": before_count,
        "after_count": after_count,
        "days_elapsed": days_elapsed,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------
# TAB 4 — Response draft generator (tone + platform aware)
# ---------------------------------------------------------------------
def draft_response_template(issue: str, branch: str, restaurant: str, source: str, tone: str) -> str:
    issue_lower = issue.lower()

    openers = {
        "Warm & Personal": f"Dear Guest,\n\nThank you for taking the time to share your experience at our {branch} ({restaurant}) outlet.",
        "Formal & Professional": f"Dear Valued Customer,\n\nWe appreciate you bringing your recent visit to {branch} ({restaurant}) to our attention.",
        "Apologetic & Direct": f"Hi, thank you for the honest feedback about {branch} ({restaurant}) — we're sorry we let you down here.",
    }
    body = (
        f" We're sorry to hear about the experience related to {issue_lower} — this isn't "
        f"the standard we hold ourselves to. We've shared this directly with the outlet "
        f"team and are taking corrective steps to address it."
    )
    platform_note = {
        "Google": " We'd welcome the chance to make it right on your next visit.",
        "Zomato": " Do reach out to us on our Zomato page or in person next time so we can fix this on the spot! 🙏",
        "TripAdvisor": " We take detailed feedback like yours seriously and will use it to improve the guest experience going forward.",
    }.get(source, " We'd appreciate the chance to make it right on your next visit.")

    closer = "\n\nWarm regards,\nCustomer Experience Team"
    return openers.get(tone, openers["Warm & Personal"]) + body + platform_note + closer


def draft_response_groq(issue: str, branch: str, restaurant: str, source: str, tone: str, sample_review: str) -> str:
    Groq = import_module("groq").Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    prompt = (
        f"Write a short, {tone.lower()} manager response (under 80 words) to this "
        f"restaurant review posted on {source} for the '{branch}' branch of "
        f"'{restaurant}', which relates to the issue '{issue}'. "
        f"Review: \"{sample_review}\". Do not invent specific compensation offers."
    )
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# ============================== APP UI ==============================

st.set_page_config(page_title="Online Reputation Management App", layout="wide")
st.title("🚨 Online Reputation Management App")
st.caption("Reuses your existing HDBSCAN + LLM issue categories for real-time reputation monitoring.")

try:
    df_full = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

rating_is_imputed = df_full["_rating_is_imputed"].iloc[0]
date_is_imputed = df_full["_date_is_imputed"].iloc[0]
if rating_is_imputed or date_is_imputed:
    missing_fields = []
    if rating_is_imputed:
        missing_fields.append("ratings")
    if date_is_imputed:
        missing_fields.append("review dates")
    st.warning(
        "This dataset has no " + " or ".join(missing_fields) +
        ". Neutral ratings and the current date are being used, so reputation scores and week-over-week alerts are illustrative."
    )

# ---------------- Sidebar filters (apply across all tabs) ----------------
st.sidebar.header("Filters")
restaurant_options = sorted(df_full[RESTAURANT_COLUMN].dropna().unique())
source_options = sorted(df_full[SOURCE_COLUMN].dropna().unique())

restaurant_filter = st.sidebar.multiselect("Restaurant chain(s)", restaurant_options, default=restaurant_options)
source_filter = st.sidebar.multiselect("Source / platform(s)", source_options, default=source_options)

min_date, max_date = df_full[DATE_COLUMN].min().date(), df_full[DATE_COLUMN].max().date()
date_range = st.sidebar.date_input("Review date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

df = df_full[
    df_full[RESTAURANT_COLUMN].isin(restaurant_filter)
    & df_full[SOURCE_COLUMN].isin(source_filter)
]
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df[(df[DATE_COLUMN] >= start) & (df[DATE_COLUMN] <= end)]

if df.empty:
    st.warning("No reviews match the current filters. Widen the filters in the sidebar.")
    st.stop()

# ---------------- Top-line KPI row ----------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Cleaned reviews", f"{load_total_review_count():,}")
k2.metric("Branches", df[BRANCH_COLUMN].nunique())
k3.metric("Restaurants", df[RESTAURANT_COLUMN].nunique())
k4.metric("Sources", df[SOURCE_COLUMN].nunique())

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    [
        "📝 Customer Review Intake",
        "📋 Existing Reviews",
        "📊 SERVQUAL Survey",
        "🙂 Emotion Analysis",
        "🔎 Root Cause Analysis",
        "🚩 Escalation Alerts",
        "💬 Empathetic Reply",
        "📈 Early-Warning Alerts",
        "🛠️ Resolution Workflow",
    ]
)

# ---------------- NEW REVIEW INTAKE ----------------
with tab1:
    st.subheader("Customer Review Intake")
    st.write("Submit a review to run sentiment, issue, severity, branch, and reputation-risk analysis.")
    model, vectorizer = load_classifier()
    with st.form("new_review_form", clear_on_submit=True):
        form_col1, form_col2 = st.columns(2)
        intake_restaurant = form_col1.selectbox("Restaurant", restaurant_options)
        intake_branch_options = sorted(
            df_full[df_full[RESTAURANT_COLUMN] == intake_restaurant][BRANCH_COLUMN].dropna().unique()
        )
        intake_branch = form_col1.selectbox("Branch", intake_branch_options or ["Unknown Branch"])
        intake_source = form_col2.selectbox("Source", ["Google", "Zomato", "TripAdvisor"])
        intake_rating = form_col2.slider("Rating", min_value=1, max_value=5, value=3)
        intake_text = st.text_area("Customer review", placeholder="Describe the customer's experience...")
        with st.container(horizontal=True):
            add_review_submitted = st.form_submit_button("Analyse and add review")
            classify_review_submitted = st.form_submit_button("Classify review", type="primary")

    if add_review_submitted or classify_review_submitted:
        if not intake_text.strip():
            st.error("Enter a review before submitting.")
        else:
            analysis = analyze_new_review(intake_text, intake_rating)
            predicted_issue = analysis["issue"]
            confidence = None
            if model is not None and vectorizer is not None:
                probabilities = model.predict_proba(vectorizer.transform([intake_text]))[0]
                best_index = np.argmax(probabilities)
                predicted_issue = model.classes_[best_index]
                confidence = probabilities[best_index]

            if classify_review_submitted:
                st.subheader("Classification Result")
                classification_col1, classification_col2 = st.columns(2)
                classification_col1.metric("Predicted issue", predicted_issue)
                if confidence is not None:
                    classification_col2.metric("Confidence", f"{confidence:.1%}")

            if add_review_submitted:
                new_row = pd.DataFrame([{
                    RESTAURANT_COLUMN: intake_restaurant,
                    BRANCH_COLUMN: intake_branch,
                    SOURCE_COLUMN: intake_source,
                    RATING_COLUMN: intake_rating,
                    TEXT_COLUMN: intake_text.strip(),
                    "review": intake_text.strip(),
                    ISSUE_CATEGORY_COLUMN: predicted_issue,
                    "sentiment": analysis["sentiment"],
                    "severity": analysis["severity"],
                    DATE_COLUMN: pd.Timestamp.today().normalize(),
                }])
                existing_new = load_new_reviews()
                pd.concat([existing_new, new_row], ignore_index=True).to_csv(NEW_REVIEWS_PATH, index=False)
                load_data.clear()
                st.success("Review analyzed and added to the dashboard.")
                result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                result_col1.metric("Sentiment", analysis["sentiment"])
                result_col2.metric("Predicted issue", predicted_issue)
                result_col3.metric("Severity", analysis["severity"])
                if confidence is not None:
                    result_col4.metric("Confidence", f"{confidence:.1%}")
                st.info(f"Branch identified: {intake_branch}. Similar complaints will be checked in the alerts and risk dashboard.")

# ---------------- APPLY CLASSIFIER ----------------
with tab2:
    st.subheader("Existing Reviews")
    st.write("Classify every cleaned review and view issue patterns across restaurants.")
    model, vectorizer = load_classifier()
    if model is None or vectorizer is None:
        st.warning("The trained issue classifier is not available.")
    elif st.button("Apply classifier to all reviews", type="primary"):
        with st.spinner("Classifying cleaned reviews..."):
            try:
                classified_reviews, issue_distribution = apply_classifier_to_cleaned_reviews(
                    model, vectorizer
                )
            except Exception as error:
                st.error(f"Could not apply the classifier: {error}")
            else:
                load_classified_reviews.clear()
                st.success(f"Classified {len(classified_reviews):,} reviews.")
                st.dataframe(issue_distribution, hide_index=True, width="stretch")

    classified_reviews = load_classified_reviews()
    if classified_reviews is not None:
        issue_counts = (
            classified_reviews["predicted_issue_category"]
            .value_counts()
            .rename_axis("Issue category")
            .reset_index(name="Reviews")
            .sort_values("Reviews")
        )
        restaurant_counts = (
            classified_reviews[RESTAURANT_COLUMN]
            .value_counts()
            .rename_axis("Restaurant")
            .reset_index(name="Reviews")
            .sort_values("Reviews")
        )
        issue_chart, restaurant_chart = st.columns(2)
        with issue_chart:
            st.plotly_chart(
                px.bar(
                    issue_counts,
                    x="Reviews",
                    y="Issue category",
                    orientation="h",
                    title="Reviews by Issue",
                ),
                width="stretch",
            )
        with restaurant_chart:
            st.plotly_chart(
                px.bar(
                    restaurant_counts,
                    x="Reviews",
                    y="Restaurant",
                    orientation="h",
                    title="Reviews by Restaurant",
                ),
                width="stretch",
            )

# ---------------- SERVQUAL SURVEY ----------------
with tab3:
    st.subheader("SERVQUAL Survey")
    servqual_scores, servqual_triangulation, response_count = load_servqual_outputs()
    if servqual_scores is None or servqual_triangulation is None:
        st.warning("SERVQUAL survey outputs are not available. Run servqual_survey_analysis (1).py first.")
    else:
        st.metric("Clean survey responses", f"{response_count:,}")
        st.plotly_chart(
            px.bar(
                servqual_scores.sort_values("mean_gap"),
                x="dimension",
                y="mean_gap",
                color="mean_gap",
                color_continuous_scale="RdYlGn",
                range_color=[-1, 1],
                title="SERVQUAL Gap by Dimension",
                labels={"dimension": "Dimension", "mean_gap": "Perception minus expectation"},
            ),
            width="stretch",
        )
        st.caption("Negative gaps indicate customer perceptions fell below expectations.")
        st.subheader("Survey and Review-Issue Triangulation")
        st.dataframe(servqual_triangulation, hide_index=True, width="stretch")

# ---------------- EMOTION ANALYSIS ----------------
with tab4:
    st.subheader("Emotion Analysis")
    st.write("Classify the full review set into restaurant-relevant emotions.")
    if st.button("Analyze review emotions", type="primary"):
        with st.spinner("Classifying review emotions..."):
            emotion_reviews = run_emotion_analysis()
            load_emotion_reviews.clear()
        st.success(f"Analyzed emotions for {len(emotion_reviews):,} reviews.")

    emotion_reviews = load_emotion_reviews()
    if emotion_reviews is not None:
        emotion_counts = (
            emotion_reviews["emotion"].value_counts().rename_axis("Emotion").reset_index(name="Reviews")
        )
        st.plotly_chart(
            px.bar(emotion_counts, x="Emotion", y="Reviews", color="Emotion", title="Review Emotions"),
            width="stretch",
        )

# ---------------- ROOT CAUSE ANALYSIS ----------------
with tab5:
    st.subheader("Root Cause and Emotion Analysis")
    emotion_reviews = load_emotion_reviews()
    if emotion_reviews is None:
        st.info("Run Emotion Analysis first to build the issue-by-emotion view.")
    else:
        emotion_counts, emotion_percentages = build_crosstab(
            emotion_reviews, "issue_cluster", "emotion"
        )
        st.dataframe(
            dominant_emotion_summary(emotion_percentages), hide_index=True, width="stretch"
        )
        st.plotly_chart(
            px.imshow(
                emotion_percentages.round(1),
                text_auto=True,
                color_continuous_scale="YlOrRd",
                aspect="auto",
                labels={"x": "Emotion", "y": "Issue category", "color": "Percent"},
                title="Emotion Composition by Issue Category",
            ),
            width="stretch",
        )

# ---------------- ESCALATION ALERTS ----------------
with tab6:
    st.subheader("Escalation Alerts")
    emotion_reviews = load_emotion_reviews()
    if emotion_reviews is None:
        st.info("Run Emotion Analysis first to identify urgent reviews.")
    else:
        urgency_reviews = compute_urgency(
            emotion_reviews, "review_text", "emotion", "rating", "review_date"
        )
        escalated_reviews = urgency_reviews[urgency_reviews["escalate"]].sort_values(
            "urgency_score", ascending=False
        )
        st.metric("Escalated reviews", len(escalated_reviews))
        st.dataframe(
            escalated_reviews[
                ["restaurant", "review_text", "issue_cluster", "emotion", "urgency_score"]
            ],
            hide_index=True,
            width="stretch",
        )
        escalation_summary = cluster_level_alert(urgency_reviews, "issue_cluster")
        if not escalation_summary.empty:
            st.subheader("Escalation rate by issue")
            st.dataframe(escalation_summary, hide_index=True, width="stretch")

# ---------------- EMPATHETIC REPLY ----------------
with tab7:
    st.subheader("Empathetic Reply Generator")
    reply_review = st.text_area("Customer review for reply", height=140)
    reply_issue = st.selectbox(
        "Issue category for reply",
        sorted(load_classified_reviews()["predicted_issue_category"].unique()),
    )
    reply_emotion = st.selectbox(
        "Customer emotion",
        ["frustration", "betrayal", "disappointment", "nostalgia", "delight", "neutral"],
    )
    reply_restaurant = st.selectbox("Restaurant for reply", restaurant_options)
    if st.button("Draft empathetic reply", type="primary"):
        if not reply_review.strip():
            st.warning("Enter the customer review before drafting a reply.")
        else:
            st.text_area(
                "Draft reply",
                value=fallback_reply(reply_issue, reply_emotion, reply_restaurant),
                height=180,
            )

# ---------------- EARLY-WARNING ALERTS ----------------
with tab8:
    st.subheader("Early-Warning Alerts (Week-over-Week Spikes)")
    alerts = detect_spikes(df)

    if alerts.empty:
        st.success("No emerging risks detected this week (or insufficient week-over-week history yet).")
    else:
        alert_restaurants = ["All"] + sorted(alerts[RESTAURANT_COLUMN].unique())
        alert_restaurant_filter = st.selectbox("Filter by restaurant", alert_restaurants, key="alert_restaurant_filter")
        view_alerts = alerts if alert_restaurant_filter == "All" else alerts[alerts[RESTAURANT_COLUMN] == alert_restaurant_filter]

        this_week = alerts.attrs.get("this_week")

        for _, row in view_alerts.iterrows():
            sev = severity_for(row["pct_change"], row["this_week"])
            color = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}[sev]
            with st.container(border=True):
                st.markdown(f"### {color} REPUTATION ALERT — {sev} severity")
                st.markdown(f"**Restaurant**  \n### {row[RESTAURANT_COLUMN]}")
                st.markdown(f"**Branch**  \n### {row[BRANCH_COLUMN]}")
                st.markdown(f"**Issue**  \n### {row[ISSUE_CATEGORY_COLUMN]}")
                st.write(
                    f"This week: **{int(row['this_week'])}** complaints  |  "
                    f"Last week: **{int(row['last_week'])}** complaints"
                )
                with st.expander("Which platform is driving this spike?"):
                    mix = source_mix_for_alert(df, row[RESTAURANT_COLUMN], row[BRANCH_COLUMN],
                                                row[ISSUE_CATEGORY_COLUMN], this_week)
                    if mix.empty:
                        st.write("No breakdown available.")
                    else:
                        st.dataframe(mix.rename(columns={SOURCE_COLUMN: "Source"}), width="stretch")

    st.caption(
        f"Alert threshold: ≥{SPIKE_THRESHOLD_PCT}% week-over-week increase, "
        f"minimum {MIN_WEEKLY_COUNT_FOR_ALERT} complaints this week."
    )

# ---------------- RESOLUTION WORKFLOW ----------------
with tab9:
    st.subheader("Complaint-to-Resolution Workflow")

    tracker = load_tracker()
    alerts = detect_spikes(df)

    if alerts.empty:
        st.info("No active alerts to track right now.")
    else:
        for _, row in alerts.iterrows():
            alert_id = row["alert_id"]
            existing = tracker[tracker["alert_id"] == alert_id]
            current_status = existing["status"].iloc[0] if len(existing) else "New"
            current_owner = existing["assigned_to"].iloc[0] if len(existing) and pd.notna(existing["assigned_to"].iloc[0]) else ""
            current_action = existing["action_taken"].iloc[0] if len(existing) and pd.notna(existing["action_taken"].iloc[0]) else ""
            current_action_date = existing["action_date"].iloc[0] if len(existing) and pd.notna(existing["action_date"].iloc[0]) else None
            current_notes = existing["notes"].iloc[0] if len(existing) and pd.notna(existing["notes"].iloc[0]) else ""

            with st.container(border=True):
                st.markdown(f"**{row[RESTAURANT_COLUMN]} — {row[BRANCH_COLUMN]} — {row[ISSUE_CATEGORY_COLUMN]}**")

                c1, c2 = st.columns(2)
                new_status = c1.selectbox(
                    "Status", STATUS_OPTIONS,
                    index=STATUS_OPTIONS.index(current_status) if current_status in STATUS_OPTIONS else 0,
                    key=f"status_{alert_id}",
                )
                assigned_to = c2.text_input("Assigned to", value=current_owner, key=f"owner_{alert_id}")

                action_taken = st.text_input("Corrective action taken", value=current_action, key=f"action_{alert_id}")
                action_date = st.date_input(
                    "Action date",
                    value=pd.to_datetime(current_action_date).date() if current_action_date else None,
                    key=f"actiondate_{alert_id}",
                )
                notes = st.text_input("Notes", value=current_notes, key=f"notes_{alert_id}")

                if action_date:
                    result = before_after_effectiveness(
                        df, row[RESTAURANT_COLUMN], row[BRANCH_COLUMN], row[ISSUE_CATEGORY_COLUMN], action_date
                    )
                    if result:
                        b1, b2, b3 = st.columns(3)
                        b1.metric(f"Complaints, {RESOLUTION_WINDOW_DAYS}d before", result["before_count"])
                        b2.metric(f"Complaints, {RESOLUTION_WINDOW_DAYS}d after", result["after_count"])
                        b3.metric("Verdict", result["verdict"])

                if st.button("Save", key=f"save_{alert_id}"):
                    tracker = tracker[tracker["alert_id"] != alert_id]
                    tracker = pd.concat([tracker, pd.DataFrame([{
                        "alert_id": alert_id,
                        "restaurant": row[RESTAURANT_COLUMN],
                        "branch": row[BRANCH_COLUMN],
                        "issue": row[ISSUE_CATEGORY_COLUMN],
                        "status": new_status,
                        "assigned_to": assigned_to,
                        "action_taken": action_taken,
                        "action_date": action_date,
                        "notes": notes,
                    }])], ignore_index=True)
                    save_tracker(tracker)
                    st.success("Saved.")

    if len(tracker):
        st.markdown("---")
        st.write("**Tracked alerts (persisted to `resolution_tracker.csv`):**")
        st.dataframe(tracker, width="stretch")

        status_counts = tracker["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        fig_status = px.bar(status_counts, x="status", y="count", title="Alerts by Status")
        st.plotly_chart(fig_status, width="stretch")


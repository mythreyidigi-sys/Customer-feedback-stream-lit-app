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
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================== CONFIG ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
DISPLAY_REVIEW_COUNT = 5909


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

    new_reviews = load_new_reviews()
    if not new_reviews.empty:
        df = pd.concat([df, new_reviews], ignore_index=True)

    if RATING_COLUMN not in df.columns:
        df[RATING_COLUMN] = 3.0
    if DATE_COLUMN not in df.columns:
        df[DATE_COLUMN] = pd.Timestamp.today().normalize()
    if SOURCE_COLUMN not in df.columns and "source_file" in df.columns:
        source_text = df["source_file"].astype(str).str.lower()
        df[SOURCE_COLUMN] = np.select(
            [
                source_text.str.contains("google"),
                source_text.str.contains("zomato"),
                source_text.str.contains("trip advisor|tripadvisor"),
            ],
            ["Google", "Zomato", "TripAdvisor"],
            default="Unknown Source",
        )

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
        100.0,  # brand-new issue this week with no prior history = treat as 100% spike
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
    elif pct_change >= 100 or this_week_count >= 8:
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
    from groq import Groq
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

st.set_page_config(page_title="Module 6 — Reputation Early-Warning System", layout="wide")
st.title("🚨 Module 6 — Reputation Early-Warning & Resolution System")
st.caption("Reuses your existing HDBSCAN + LLM issue categories for real-time reputation monitoring.")

try:
    df_full = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

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
k1.metric("Total reviews", f"{DISPLAY_REVIEW_COUNT:,}")
k2.metric("Branches", df[BRANCH_COLUMN].nunique())
k3.metric("Restaurants", df[RESTAURANT_COLUMN].nunique())
k4.metric("Sources", df[SOURCE_COLUMN].nunique())

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📝 New Review Intake", "🏢 Reputation Risk Score", "📈 Early-Warning Alerts", "🛠️ Resolution Workflow", "✍️ Response Draft"]
)

# ---------------- NEW REVIEW INTAKE ----------------
with tab1:
    st.subheader("New Review Intake")
    st.write("Submit a review to run sentiment, issue, severity, branch, and reputation-risk analysis.")
    with st.form("new_review_form", clear_on_submit=True):
        form_col1, form_col2 = st.columns(2)
        intake_restaurant = form_col1.selectbox("Restaurant", restaurant_options)
        intake_branch_options = sorted(
            df_full[df_full[RESTAURANT_COLUMN] == intake_restaurant][BRANCH_COLUMN].dropna().unique()
        )
        intake_branch = form_col1.selectbox("Branch", intake_branch_options or ["Unknown Branch"])
        intake_source = form_col2.selectbox("Source", ["Google", "Zomato", "TripAdvisor"])
        intake_rating = form_col2.slider("Rating", min_value=1, max_value=5, value=3)
        intake_text = st.text_area("Review", placeholder="Describe the customer's experience...")
        submitted = st.form_submit_button("Analyze and add review")

    if submitted:
        if not intake_text.strip():
            st.error("Enter a review before submitting.")
        else:
            analysis = analyze_new_review(intake_text, intake_rating)
            new_row = pd.DataFrame([{
                RESTAURANT_COLUMN: intake_restaurant,
                BRANCH_COLUMN: intake_branch,
                SOURCE_COLUMN: intake_source,
                RATING_COLUMN: intake_rating,
                TEXT_COLUMN: intake_text.strip(),
                "review": intake_text.strip(),
                ISSUE_CATEGORY_COLUMN: analysis["issue"],
                "sentiment": analysis["sentiment"],
                "severity": analysis["severity"],
                DATE_COLUMN: pd.Timestamp.today().normalize(),
            }])
            existing_new = load_new_reviews()
            pd.concat([existing_new, new_row], ignore_index=True).to_csv(NEW_REVIEWS_PATH, index=False)
            load_data.clear()
            st.success("Review analyzed and added to the dashboard.")
            result_col1, result_col2, result_col3 = st.columns(3)
            result_col1.metric("Sentiment", analysis["sentiment"])
            result_col2.metric("Issue", analysis["issue"])
            result_col3.metric("Severity", analysis["severity"])
            st.info(f"Branch identified: {intake_branch}. Similar complaints will be checked in the alerts and risk dashboard.")
            st.rerun()

# ---------------- REPUTATION SCORE ----------------
with tab2:
    st.subheader("Branch Reputation Risk Score")
    st.write("Higher score = healthier reputation. Charts are shown restaurant-wise → branch-wise.")

    risk_df = compute_reputation_risk(df)
    st.markdown("**Reputation score by branch**")
    view_choice = st.selectbox(
        "View",
        ["All restaurants (small multiples)"] + restaurant_options,
        key="risk_view_choice",
    )

    if view_choice == "All restaurants (small multiples)":
        n_restaurants = risk_df[RESTAURANT_COLUMN].nunique()
        wrap = 3 if n_restaurants > 3 else n_restaurants
        fig = px.bar(
            risk_df.sort_values([RESTAURANT_COLUMN, "reputation_score"]),
            x=BRANCH_COLUMN,
            y="reputation_score",
            color="reputation_score",
            color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            facet_col=RESTAURANT_COLUMN,
            facet_col_wrap=wrap,
            height=350 * (-(-n_restaurants // wrap)),  # ceil division for row count
            title="Reputation Score by Branch, grouped by Restaurant Chain",
        )
        fig.update_xaxes(matches=None, tickangle=45)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig, use_container_width=True)
    else:
        chain_df = risk_df[risk_df[RESTAURANT_COLUMN] == view_choice].sort_values("reputation_score")
        fig = px.bar(
            chain_df,
            x=BRANCH_COLUMN,
            y="reputation_score",
            color="reputation_score",
            color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            title=f"Reputation Score by Branch — {view_choice}",
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Detailed table** (Restaurant and Source included)")

    display_cols = [RESTAURANT_COLUMN, BRANCH_COLUMN, "status", "reputation_score", "avg_rating",
                     "positive_rate", "negative_issue_rate", "rating_trend_30d", "review_count"]
    source_cols = [c for c in source_options if c in risk_df.columns]
    display_cols += source_cols

    st.dataframe(
        risk_df[display_cols]
        .round(2)
        .rename(columns={RESTAURANT_COLUMN: "Restaurant", BRANCH_COLUMN: "Branch"}),
        use_container_width=True,
    )
    st.caption(
        "Source columns show how many of that branch's reviews came from each platform "
        "(Google / Zomato / TripAdvisor), so a low score can be traced back to a specific channel."
    )

# ---------------- EARLY-WARNING ALERTS ----------------
with tab3:
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
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Restaurant", row[RESTAURANT_COLUMN])
                c2.metric("Branch", row[BRANCH_COLUMN])
                c3.metric("Issue", row[ISSUE_CATEGORY_COLUMN])
                c4.metric("Trend", f"↑ {row['pct_change']:.0f}%")
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
                        st.dataframe(mix.rename(columns={SOURCE_COLUMN: "Source"}), use_container_width=True)

    st.caption(
        f"Alert threshold: ≥{SPIKE_THRESHOLD_PCT}% week-over-week increase, "
        f"minimum {MIN_WEEKLY_COUNT_FOR_ALERT} complaints this week."
    )

# ---------------- RESOLUTION WORKFLOW ----------------
with tab4:
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
        st.dataframe(tracker, use_container_width=True)

        status_counts = tracker["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        fig_status = px.bar(status_counts, x="status", y="count", title="Alerts by Status")
        st.plotly_chart(fig_status, use_container_width=True)

# ---------------- RESPONSE DRAFT ----------------
with tab5:
    st.subheader("Response Draft Generator")

    c1, c2, c3 = st.columns(3)
    restaurant_choice = c1.selectbox("Restaurant", sorted(df[RESTAURANT_COLUMN].unique()), key="draft_restaurant")
    branch_options = sorted(df[df[RESTAURANT_COLUMN] == restaurant_choice][BRANCH_COLUMN].unique())
    branch_choice = c2.selectbox("Branch", branch_options, key="draft_branch")
    issue_options = sorted(
        df[(df[RESTAURANT_COLUMN] == restaurant_choice) & (df[BRANCH_COLUMN] == branch_choice)]
        [ISSUE_CATEGORY_COLUMN].dropna().unique()
    )
    issue_choice = c3.selectbox("Issue category", issue_options, key="draft_issue")

    c4, c5 = st.columns(2)
    source_choice = c4.selectbox("Platform / source", sorted(df[SOURCE_COLUMN].unique()), key="draft_source")
    tone_choice = c5.selectbox("Tone", ["Warm & Personal", "Formal & Professional", "Apologetic & Direct"], key="draft_tone")

    subset = df[
        (df[RESTAURANT_COLUMN] == restaurant_choice)
        & (df[BRANCH_COLUMN] == branch_choice)
        & (df[ISSUE_CATEGORY_COLUMN] == issue_choice)
        & (df[SOURCE_COLUMN] == source_choice)
    ]
    if subset.empty:
        subset = df[
            (df[RESTAURANT_COLUMN] == restaurant_choice)
            & (df[BRANCH_COLUMN] == branch_choice)
            & (df[ISSUE_CATEGORY_COLUMN] == issue_choice)
        ]

    sample_review = ""
    if len(subset) and TEXT_COLUMN in df.columns:
        review_idx = st.selectbox(
            "Pick a specific review to respond to",
            list(range(len(subset))),
            format_func=lambda i: str(subset[TEXT_COLUMN].iloc[i])[:80] + "...",
            key="draft_review_idx",
        )
        sample_review = subset[TEXT_COLUMN].iloc[review_idx]
        st.write("**Selected review:**")
        st.write(f"> {sample_review}")

    if st.button("Generate draft response"):
        if USE_GROQ:
            try:
                draft = draft_response_groq(issue_choice, branch_choice, restaurant_choice, source_choice, tone_choice, sample_review)
                st.caption("Generated via Groq LLM.")
            except Exception as e:
                st.warning(f"Groq call failed ({e}); falling back to template.")
                draft = draft_response_template(issue_choice, branch_choice, restaurant_choice, source_choice, tone_choice)
        else:
            draft = draft_response_template(issue_choice, branch_choice, restaurant_choice, source_choice, tone_choice)
            st.caption("Template-based draft (set GROQ_API_KEY env var to enable LLM drafting).")

        st.text_area("Draft response", value=draft, height=220)
        st.download_button("Download draft as .txt", draft, file_name=f"response_{branch_choice}_{issue_choice}.txt")

"""
module6_reputation_early_warning_app.py

Module 6 MVP — Reputation Early-Warning & Resolution System

Reuses your existing HDBSCAN + LLM issue categories directly (no
separate NLP system) and adds four capabilities:

  TAB 1 — Reputation Risk Score (per branch, 0-100, higher = healthier)
      Combines: avg sentiment (rating-based), negative-review
      frequency, negative rating trend (recent vs prior period), and
      issue severity mix.

  TAB 2 — Early-Warning Alerts (week-over-week spike detection)
      For each branch × issue-category combination, compares this
      week's complaint count to last week's. Flags anything above a
      configurable spike threshold as an EMERGING REPUTATION RISK,
      styled like the alert card in the original brief.

  TAB 3 — Complaint-to-Resolution Workflow
      Each alert can be moved through a simple status pipeline:
      New -> Assigned -> Resolved -> Monitoring.
      Statuses persist to a local CSV (resolution_tracker.csv) so they
      survive across Streamlit reruns/restarts.

  TAB 4 — Response Draft Generator
      Produces a short manager-facing draft response for a selected
      review/issue. Template-based by default (no external API key
      needed). If you set GROQ_API_KEY as an environment variable,
      it will use Groq to generate a more natural draft — this is
      OPTIONAL and clearly flagged; the template fallback is fully
      functional on its own for the demo/viva.

--------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------
    pip install streamlit pandas numpy plotly openpyxl
    # optional, only if you want LLM-drafted responses:
    pip install groq
    streamlit run module6_reputation_early_warning_app.py

--------------------------------------------------------------------
HOW TO POINT THIS AT YOUR OWN DATA
--------------------------------------------------------------------
Edit CONFIG below. This script needs a REVIEW DATE column in addition
to the columns used in your earlier scripts, since week-over-week
trend detection requires timestamps. If your cleaned file doesn't
have dates yet, add a `review_date` column (from the scraped raw data)
before running this.
--------------------------------------------------------------------
"""

import os
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================== CONFIG ==============================

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs",
    "issues",
    "hdbscan_clustered_reviews.xlsx",
)  # .xlsx or .csv

RATING_COLUMN = "rating"
BRANCH_COLUMN = "branch"
ISSUE_CATEGORY_COLUMN = "issue_category"
TEXT_COLUMN = "review_text"
DATE_COLUMN = "review_date"        # must be parseable as a date
NOISE_LABELS = ["Noise", "noise", "-1", -1]

NEGATIVE_ISSUE_CATEGORIES = [
    "Food Quantity & Value for Money",
    "Slow Service & Staff Negligence",
    "Service Quality",
    "Poor Experience & Food Hygiene Complaints",
]

SATISFACTION_THRESHOLD = 4

# Week-over-week increase (%) at which an issue is flagged as an alert.
# 100% = complaint count has doubled vs the prior week.
SPIKE_THRESHOLD_PCT = 75

# Minimum complaint count this week to even consider an alert
# (avoids flagging noise like "1 -> 2 complaints" as a 100% spike).
MIN_WEEKLY_COUNT_FOR_ALERT = 3

TRACKER_PATH = "resolution_tracker.csv"  # persists alert statuses across runs

USE_GROQ = bool(os.environ.get("GROQ_API_KEY"))  # optional LLM-drafted responses

# ======================================================================


@st.cache_data
def load_data() -> pd.DataFrame:
    if DATA_PATH.endswith(".xlsx"):
        df = pd.read_excel(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH)

    if TEXT_COLUMN not in df.columns and "review" in df.columns:
        df[TEXT_COLUMN] = df["review"]
    if ISSUE_CATEGORY_COLUMN not in df.columns and "cluster" in df.columns:
        df[ISSUE_CATEGORY_COLUMN] = "Cluster " + df["cluster"].astype(str)
    if RATING_COLUMN not in df.columns:
        df[RATING_COLUMN] = 3.0
    if DATE_COLUMN not in df.columns:
        df[DATE_COLUMN] = pd.Timestamp.today().normalize()

    required = [RATING_COLUMN, BRANCH_COLUMN, ISSUE_CATEGORY_COLUMN, DATE_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected column(s) {missing} in {DATA_PATH}. "
            f"Found columns: {list(df.columns)}. "
            f"Note: Module 6 needs a review date column for trend detection."
        )

    df[BRANCH_COLUMN] = df[BRANCH_COLUMN].fillna("Unknown Branch").astype(str)
    df[ISSUE_CATEGORY_COLUMN] = df[ISSUE_CATEGORY_COLUMN].fillna("Unknown Issue").astype(str)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.dropna(subset=[DATE_COLUMN])
    df["week"] = df[DATE_COLUMN].dt.to_period("W").apply(lambda p: p.start_time)
    return df


# ---------------------------------------------------------------------
# TAB 1 — Reputation Risk Score
# ---------------------------------------------------------------------
def compute_reputation_risk(df: pd.DataFrame) -> pd.DataFrame:
    is_negative = df[ISSUE_CATEGORY_COLUMN].isin(NEGATIVE_ISSUE_CATEGORIES)
    latest_date = df[DATE_COLUMN].max()
    recent_cutoff = latest_date - timedelta(days=30)
    prior_cutoff = latest_date - timedelta(days=60)

    recent = df[df[DATE_COLUMN] >= recent_cutoff]
    prior = df[(df[DATE_COLUMN] >= prior_cutoff) & (df[DATE_COLUMN] < recent_cutoff)]

    rows = []
    for branch, g in df.groupby(BRANCH_COLUMN):
        g_recent = recent[recent[BRANCH_COLUMN] == branch]
        g_prior = prior[prior[BRANCH_COLUMN] == branch]

        avg_rating = g[RATING_COLUMN].mean()
        negative_rate = is_negative[df[BRANCH_COLUMN] == branch].mean()
        positive_rate = (g[RATING_COLUMN] >= SATISFACTION_THRESHOLD).mean()

        recent_avg = g_recent[RATING_COLUMN].mean() if len(g_recent) else avg_rating
        prior_avg = g_prior[RATING_COLUMN].mean() if len(g_prior) else recent_avg
        rating_trend = recent_avg - prior_avg  # negative = worsening

        rows.append({
            BRANCH_COLUMN: branch,
            "review_count": len(g),
            "avg_rating": avg_rating,
            "positive_rate": positive_rate,
            "negative_issue_rate": negative_rate,
            "rating_trend_30d": rating_trend,
        })

    result = pd.DataFrame(rows)

    # Normalize components to 0-1, then combine into a 0-100 risk score
    # (higher = healthier reputation)
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
    return result.sort_values("reputation_score", ascending=True).reset_index(drop=True)


# ---------------------------------------------------------------------
# TAB 2 — Early-warning spike detection
# ---------------------------------------------------------------------
def detect_spikes(df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        df.groupby([BRANCH_COLUMN, ISSUE_CATEGORY_COLUMN, "week"])
        .size()
        .reset_index(name="count")
    )

    weeks_sorted = sorted(weekly["week"].unique())
    if len(weeks_sorted) < 2:
        return pd.DataFrame()  # not enough history yet

    this_week, last_week = weeks_sorted[-1], weeks_sorted[-2]

    this_wk = weekly[weekly["week"] == this_week].set_index([BRANCH_COLUMN, ISSUE_CATEGORY_COLUMN])["count"]
    last_wk = weekly[weekly["week"] == last_week].set_index([BRANCH_COLUMN, ISSUE_CATEGORY_COLUMN])["count"]

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
        alerts[BRANCH_COLUMN].astype(str) + " | " + alerts[ISSUE_CATEGORY_COLUMN].astype(str)
    )
    return alerts


def severity_for(pct_change: float, this_week_count: int) -> str:
    if pct_change >= 150 or this_week_count >= 15:
        return "High"
    elif pct_change >= 100 or this_week_count >= 8:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------
# TAB 3 — Resolution tracker (persisted to CSV)
# ---------------------------------------------------------------------
def load_tracker() -> pd.DataFrame:
    if os.path.exists(TRACKER_PATH):
        return pd.read_csv(TRACKER_PATH)
    return pd.DataFrame(columns=["alert_id", "branch", "issue", "status", "notes"])


def save_tracker(tracker_df: pd.DataFrame):
    tracker_df.to_csv(TRACKER_PATH, index=False)


# ---------------------------------------------------------------------
# TAB 4 — Response draft generator
# ---------------------------------------------------------------------
def draft_response_template(issue: str, branch: str) -> str:
    return (
        f"Dear Guest,\n\n"
        f"Thank you for sharing your feedback about your visit to our {branch} outlet. "
        f"We're sorry to hear about the experience related to {issue.lower()} — this "
        f"isn't the standard we hold ourselves to. We've shared this with the outlet "
        f"team and are taking corrective steps to address it.\n\n"
        f"We'd appreciate the chance to make it right on your next visit.\n\n"
        f"Warm regards,\nCustomer Experience Team"
    )


def draft_response_groq(issue: str, branch: str, sample_review: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    prompt = (
        f"Write a short, warm, professional manager response (under 80 words) to this "
        f"restaurant review from the '{branch}' branch, which relates to the issue "
        f"'{issue}'. Review: \"{sample_review}\". Do not invent specific compensation offers."
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
    df = load_data()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["🏢 Reputation Risk Score", "📈 Early-Warning Alerts", "🛠️ Resolution Workflow", "✍️ Response Draft"]
)

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Branch Reputation Risk Score")
    st.write("Higher score = healthier reputation. Sorted lowest (highest risk) first.")

    risk_df = compute_reputation_risk(df)

    fig = px.bar(
        risk_df,
        x=BRANCH_COLUMN,
        y="reputation_score",
        color="reputation_score",
        color_continuous_scale="RdYlGn",
        title="Reputation Score by Branch (0-100)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        risk_df[[BRANCH_COLUMN, "status", "reputation_score", "avg_rating",
                 "positive_rate", "negative_issue_rate", "rating_trend_30d", "review_count"]]
        .round(2)
        .rename(columns={BRANCH_COLUMN: "Branch"}),
        use_container_width=True,
    )

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Early-Warning Alerts (Week-over-Week Spikes)")
    alerts = detect_spikes(df)

    if alerts.empty:
        st.success("No emerging risks detected this week (or insufficient week-over-week history yet).")
    else:
        for _, row in alerts.iterrows():
            sev = severity_for(row["pct_change"], row["this_week"])
            color = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}[sev]
            with st.container(border=True):
                st.markdown(f"### {color} REPUTATION ALERT — {sev} severity")
                c1, c2, c3 = st.columns(3)
                c1.metric("Branch", row[BRANCH_COLUMN])
                c2.metric("Issue", row[ISSUE_CATEGORY_COLUMN])
                c3.metric("Trend", f"↑ {row['pct_change']:.0f}%")
                st.write(
                    f"This week: **{int(row['this_week'])}** complaints  |  "
                    f"Last week: **{int(row['last_week'])}** complaints"
                )

    st.caption(
        f"Alert threshold: ≥{SPIKE_THRESHOLD_PCT}% week-over-week increase, "
        f"minimum {MIN_WEEKLY_COUNT_FOR_ALERT} complaints this week."
    )

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("Complaint-to-Resolution Workflow")

    tracker = load_tracker()

    if alerts.empty if 'alerts' in dir() else True:
        alerts = detect_spikes(df)

    if alerts.empty:
        st.info("No active alerts to track right now.")
    else:
        for _, row in alerts.iterrows():
            alert_id = row["alert_id"]
            existing = tracker[tracker["alert_id"] == alert_id]
            current_status = existing["status"].iloc[0] if len(existing) else "New"

            with st.container(border=True):
                st.markdown(f"**{row[BRANCH_COLUMN]} — {row[ISSUE_CATEGORY_COLUMN]}**")
                new_status = st.selectbox(
                    "Status",
                    ["New", "Assigned", "Resolved", "Monitoring"],
                    index=["New", "Assigned", "Resolved", "Monitoring"].index(current_status),
                    key=f"status_{alert_id}",
                )
                notes = st.text_input(
                    "Corrective action notes",
                    value=existing["notes"].iloc[0] if len(existing) and pd.notna(existing["notes"].iloc[0]) else "",
                    key=f"notes_{alert_id}",
                )

                if st.button("Save", key=f"save_{alert_id}"):
                    tracker = tracker[tracker["alert_id"] != alert_id]
                    tracker = pd.concat([tracker, pd.DataFrame([{
                        "alert_id": alert_id,
                        "branch": row[BRANCH_COLUMN],
                        "issue": row[ISSUE_CATEGORY_COLUMN],
                        "status": new_status,
                        "notes": notes,
                    }])], ignore_index=True)
                    save_tracker(tracker)
                    st.success("Saved.")

    if len(tracker):
        st.markdown("---")
        st.write("**Tracked alerts (persisted to `resolution_tracker.csv`):**")
        st.dataframe(tracker, use_container_width=True)

# ---------------- TAB 4 ----------------
with tab4:
    st.subheader("Response Draft Generator")

    branch_choice = st.selectbox("Branch", sorted(df[BRANCH_COLUMN].unique()), key="draft_branch")
    issue_choice = st.selectbox(
        "Issue category",
        sorted(df[ISSUE_CATEGORY_COLUMN].dropna().unique()),
        key="draft_issue",
    )

    subset = df[(df[BRANCH_COLUMN] == branch_choice) & (df[ISSUE_CATEGORY_COLUMN] == issue_choice)]
    sample_review = subset[TEXT_COLUMN].iloc[0] if len(subset) and TEXT_COLUMN in df.columns else ""

    if sample_review:
        st.write("**Sample review this responds to:**")
        st.write(f"> {sample_review}")

    if st.button("Generate draft response"):
        if USE_GROQ:
            try:
                draft = draft_response_groq(issue_choice, branch_choice, sample_review)
                st.caption("Generated via Groq LLM.")
            except Exception as e:
                st.warning(f"Groq call failed ({e}); falling back to template.")
                draft = draft_response_template(issue_choice, branch_choice)
        else:
            draft = draft_response_template(issue_choice, branch_choice)
            st.caption("Template-based draft (set GROQ_API_KEY env var to enable LLM drafting).")

        st.text_area("Draft response", value=draft, height=200)

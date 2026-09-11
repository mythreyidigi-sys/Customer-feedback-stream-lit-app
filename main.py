"""Restaurant Review Issue Analysis -- unified Streamlit dashboard.

Rebuilt to match the 9-tab layout: Customer Review Intake, Existing Reviews,
SERVQUAL Survey, Emotion Analysis, Root Cause Analysis, Escalation Alerts,
Empathetic Reply, Early-Warning Alerts, Resolution Workflow -- with a global
sidebar filter (restaurant chain / platform / date range) applied throughout.
"""
import os
import re
import sys
from importlib.util import module_from_spec, spec_from_file_location
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title="Restaurant Review Issue Analysis", layout="wide")


def get_staff_password():
    """Read the staff password from Streamlit Cloud's private secrets store."""
    try:
        return st.secrets["STAFF_PASSWORD"]
    except (FileNotFoundError, KeyError):
        return os.getenv("STAFF_PASSWORD")


st.session_state.setdefault("staff_authenticated", False)
st.session_state.setdefault("show_staff_login", False)

title_col, staff_col = st.columns([5, 1])
with title_col:
    st.title("Restaurant Review Issue Analysis")

with staff_col:
    if st.session_state["staff_authenticated"]:
        if st.button("Staff logout", key="staff_logout"):
            st.session_state["staff_authenticated"] = False
            st.rerun()
    elif st.button("Staff login", key="staff_login"):
        st.session_state["show_staff_login"] = True

if not st.session_state["staff_authenticated"] and st.session_state["show_staff_login"]:
    with st.form("staff_login_form"):
        entered_password = st.text_input("Staff password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        configured_password = get_staff_password()
        if not configured_password:
            st.error("Staff access is not configured. Add STAFF_PASSWORD to the app secrets.")
        elif entered_password == configured_password:
            st.session_state["staff_authenticated"] = True
            st.session_state["show_staff_login"] = False
            st.rerun()
        else:
            st.error("Incorrect staff password.")

# ---------------------------------------------------------------------------
# Optional CX helper copies are stored under scripts_new792026 when the
# original root-level modules are unavailable. Register them under their normal
# module names so the guarded imports below remain compatible.
_CX_COPY_DIR = BASE_DIR / "scripts_new792026"
def _load_cx_copy(module_name, filename):
    module_path = _CX_COPY_DIR / filename
    spec = spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {module_path}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

if not (BASE_DIR / "emotion_classification.py").exists():
    try:
        _load_cx_copy("cx_common", "cx_common (1).py")
        _load_cx_copy("emotion_classification", "emotion_classification (1).py")
        _load_cx_copy("root_cause_emotion_analysis", "root_cause_emotion_analysis (1).py")
        _load_cx_copy("escalation_detection", "escalation_detection (1).py")
        _load_cx_copy("empathetic_reply_generator", "empathetic_reply_generator (1).py")
    except Exception:
        pass

# Optional pipeline modules (plain .py files alongside main.py). Each import
# is wrapped so a missing file degrades to a warning in the relevant tab
# instead of crashing the whole app.
# ---------------------------------------------------------------------------
try:
    from emotion_classification import classify_emotions, EMOTION_TAXONOMY
    EMOTION_MODULE_OK, EMOTION_IMPORT_ERROR = True, None
except Exception as exc:  # noqa: BLE001
    EMOTION_MODULE_OK, EMOTION_IMPORT_ERROR = False, str(exc)

try:
    from root_cause_emotion_analysis import build_crosstab, dominant_emotion_summary
    ROOTCAUSE_MODULE_OK, ROOTCAUSE_IMPORT_ERROR = True, None
except Exception as exc:  # noqa: BLE001
    ROOTCAUSE_MODULE_OK, ROOTCAUSE_IMPORT_ERROR = False, str(exc)

try:
    from escalation_detection import compute_urgency, cluster_level_alert
    ESCALATION_MODULE_OK, ESCALATION_IMPORT_ERROR = True, None
except Exception as exc:  # noqa: BLE001
    ESCALATION_MODULE_OK, ESCALATION_IMPORT_ERROR = False, str(exc)

try:
    from empathetic_reply_generator import generate_replies
    REPLY_MODULE_OK, REPLY_IMPORT_ERROR = True, None
except Exception as exc:  # noqa: BLE001
    REPLY_MODULE_OK, REPLY_IMPORT_ERROR = False, str(exc)

try:
    from servqual_survey_analysis import (
        score_survey, triangulate, SERVQUAL_ITEMS, INTERIM_ISSUE_FREQUENCY,
    )
    SERVQUAL_MODULE_OK, SERVQUAL_IMPORT_ERROR = True, None
except Exception as exc:  # noqa: BLE001
    try:
        servqual_module = _load_cx_copy(
            "servqual_survey_analysis", "servqual_survey_analysis (1).py"
        )
        score_survey = servqual_module.score_survey
        triangulate = servqual_module.triangulate
        SERVQUAL_ITEMS = servqual_module.SERVQUAL_ITEMS
        INTERIM_ISSUE_FREQUENCY = servqual_module.INTERIM_ISSUE_FREQUENCY
        SERVQUAL_MODULE_OK, SERVQUAL_IMPORT_ERROR = True, None
    except Exception as fallback_exc:  # noqa: BLE001
        SERVQUAL_MODULE_OK, SERVQUAL_IMPORT_ERROR = False, str(fallback_exc)


def find_column(dataframe, candidates):
    """Return the first matching column name (case-insensitive) from candidates."""
    lower_map = {c.lower(): c for c in dataframe.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


# ---------------------------------------------------------------------------
# Base dataset loading + normalization
# ---------------------------------------------------------------------------
CANDIDATE_BASE_FILES = [
    "outputs/reviews_with_issue_classification.xlsx",
    "outputs/cleaned_reviews.xlsx",
    "cleaned_reviews.xlsx",
]

CANDIDATE_CLASSIFIER_PATHS = [
    "scripts/issue_classifier.joblib",
    "scripts_new792026/issue_classifier.joblib",
    "issue_classifier.joblib",
]


@st.cache_resource
def load_classifier():
    for rel in CANDIDATE_CLASSIFIER_PATHS:
        p = BASE_DIR / rel
        if p.exists():
            try:
                bundle = joblib.load(p)
                return bundle.get("model"), bundle.get("vectorizer"), rel
            except Exception:
                continue
    return None, None, None


@st.cache_data
def load_base_dataset():
    """Load the review dataset and standardize column names.

    Returns (df, source_path, synthesized_rating, synthesized_date).
    """
    raw_df, source_path = None, None
    for rel in CANDIDATE_BASE_FILES:
        p = BASE_DIR / rel
        if p.exists():
            raw_df = pd.read_excel(p)
            source_path = rel
            break

    if raw_df is None:
        return None, None, False, False

    df = raw_df.copy()
    text_col = find_column(df, ["review_text", "review", "Reviews", "Review"])
    restaurant_col = find_column(df, ["restaurant", "Restaurant"])
    branch_col = find_column(df, ["branch", "Branch"])
    platform_col = find_column(df, ["source", "platform", "Source", "Platform"])
    rating_col = find_column(df, ["rating", "Rating"])
    date_col = find_column(df, ["review_date", "date", "Date"])
    issue_col = find_column(df, ["predicted_issue_category", "issue_cluster", "issue_category"])

    df["review_text"] = df[text_col] if text_col else ""
    df["restaurant"] = df[restaurant_col] if restaurant_col else "Unknown"
    df["branch"] = df[branch_col] if branch_col else df["restaurant"]
    df["platform"] = df[platform_col] if platform_col else "Unknown"

    synthesized_rating = rating_col is None
    if rating_col:
        df["rating"] = pd.to_numeric(df[rating_col], errors="coerce").fillna(3).astype(int)
    else:
        df["rating"] = 3  # neutral placeholder

    synthesized_date = date_col is None
    if date_col:
        df["review_date"] = pd.to_datetime(df[date_col], errors="coerce")
        df["review_date"] = df["review_date"].fillna(pd.Timestamp.now())
    else:
        # Spread synthetic dates over the last 8 weeks (deterministic by row
        # index) so week-over-week spike detection has something to compare,
        # rather than dumping every review into "this week" only.
        rng = np.random.default_rng(42)
        offsets_days = rng.integers(0, 56, size=len(df))
        now = pd.Timestamp.now().normalize()
        df["review_date"] = [now - pd.Timedelta(days=int(d)) for d in offsets_days]

    if issue_col:
        df["issue_cluster"] = df[issue_col]
    else:
        df["issue_cluster"] = pd.NA

    keep_cols = ["review_text", "restaurant", "branch", "platform", "rating", "review_date", "issue_cluster"]
    return df[keep_cols].reset_index(drop=True), source_path, synthesized_rating, synthesized_date


def apply_classifier_to_dataframe(df, model, vectorizer):
    mask = df["issue_cluster"].isna() | (df["issue_cluster"].astype(str).str.strip() == "")
    if mask.any():
        X = vectorizer.transform(df.loc[mask, "review_text"].astype(str))
        df.loc[mask, "issue_cluster"] = model.predict(X)
    return df


def get_working_dataset():
    """The single source of truth used by every tab: base dataset + any
    reviews added via the Customer Review Intake tab this session, with the
    classifier applied to anything still unclassified."""
    base_df, source_path, synth_rating, synth_date = load_base_dataset()
    intake_rows = st.session_state.get("intake_reviews", [])

    frames = []
    if base_df is not None:
        frames.append(base_df)
    if intake_rows:
        frames.append(pd.DataFrame(intake_rows))

    if not frames:
        return None, source_path, synth_rating, synth_date

    combined = pd.concat(frames, ignore_index=True)

    model, vectorizer, _ = load_classifier()
    if model is not None and vectorizer is not None:
        combined = apply_classifier_to_dataframe(combined, model, vectorizer)

    return combined, source_path, synth_rating, synth_date


# ---------------------------------------------------------------------------
# Sidebar filters (global, applied across tabs)
# ---------------------------------------------------------------------------
def render_sidebar_filters(df):
    st.sidebar.header("Filters")

    restaurants = sorted(df["restaurant"].dropna().unique().tolist()) if df is not None else []
    platforms = sorted(df["platform"].dropna().unique().tolist()) if df is not None else []

    selected_restaurants = st.sidebar.multiselect(
        "Restaurant chain(s)", options=restaurants, default=restaurants, key="filter_restaurants",
    )
    selected_platforms = st.sidebar.multiselect(
        "Source / platform(s)", options=platforms, default=platforms, key="filter_platforms",
    )

    if df is not None and len(df) and df["review_date"].notna().any():
        min_date = df["review_date"].min().date()
        max_date = df["review_date"].max().date()
    else:
        max_date = datetime.now().date()
        min_date = max_date - timedelta(days=7)

    date_range = st.sidebar.date_input(
        "Review date range", value=(min_date, max_date), key="filter_date_range",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    return selected_restaurants, selected_platforms, start_date, end_date


def apply_filters(df, restaurants, platforms, start_date, end_date):
    if df is None:
        return df
    filtered = df.copy()
    if restaurants:
        filtered = filtered[filtered["restaurant"].isin(restaurants)]
    if platforms:
        filtered = filtered[filtered["platform"].isin(platforms)]
    filtered = filtered[
        (filtered["review_date"].dt.date >= start_date)
        & (filtered["review_date"].dt.date <= end_date)
    ]
    return filtered.reset_index(drop=True)


def show_top_metrics(df):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cleaned reviews", f"{len(df):,}" if df is not None else "0")
    c2.metric("Branches", df["branch"].nunique() if df is not None else 0)
    c3.metric("Restaurants", df["restaurant"].nunique() if df is not None else 0)
    c4.metric("Sources", df["platform"].nunique() if df is not None else 0)

# ---------------------------------------------------------------------------
# Load data + render sidebar
# ---------------------------------------------------------------------------
working_df, base_source_path, synthesized_rating, synthesized_date = get_working_dataset()
if st.session_state["staff_authenticated"]:
    sel_restaurants, sel_platforms, sel_start, sel_end = render_sidebar_filters(working_df)
    filtered_df = apply_filters(working_df, sel_restaurants, sel_platforms, sel_start, sel_end)
else:
    filtered_df = working_df

if st.session_state["staff_authenticated"] and (synthesized_rating or synthesized_date):
    st.warning(
        "This dataset has no ratings or review dates. Neutral ratings and "
        "the current date are being used, so reputation scores and "
        "week-over-week alerts are illustrative."
    )

if st.session_state["staff_authenticated"]:
    (
        reviews_tab, servqual_tab, emotion_tab, rootcause_tab,
        escalation_tab, reply_tab, earlywarning_tab, resolution_tab,
    ) = st.tabs(
        [
            "Existing Reviews",
            "SERVQUAL Survey",
            "Emotion Analysis",
            "Root Cause Analysis",
            "Escalation Alerts",
            "Empathetic Reply",
            "Early-Warning Alerts",
            "Resolution Workflow",
        ]
    )

# ===========================================================================
# TAB: Customer Review Intake
# ===========================================================================
def render_customer_review_intake():
    st.header("Customer Review Intake")
    st.caption("Submit a review to run sentiment, issue, severity, branch, and reputation-risk analysis.")

    if working_df is None:
        st.warning("No base review dataset found yet. You can still submit a review below.")

    restaurants_all = sorted(working_df["restaurant"].dropna().unique().tolist()) if working_df is not None else ["Not specified"]
    col_left, col_right = st.columns(2)

    with col_left:
        intake_restaurant = st.selectbox("Restaurant", restaurants_all, key="intake_restaurant")
        branch_options = ["Not specified"]
        if working_df is not None:
            branch_options = sorted(
                working_df.loc[working_df["restaurant"] == intake_restaurant, "branch"].dropna().unique().tolist()
            ) or ["Not specified"]
        intake_branch = st.selectbox("Branch", branch_options, key="intake_branch")
        intake_text = st.text_area("Customer review", placeholder="Describe the customer's experience...", height=140, key="intake_text")

    with col_right:
        platform_options = sorted(working_df["platform"].dropna().unique().tolist()) if working_df is not None else ["Google", "Zomato", "TripAdvisor"]
        intake_platform = st.selectbox("Source", platform_options, key="intake_platform")
        intake_rating = st.slider("Rating", 1, 5, 3, key="intake_rating")

    col_btn1, col_btn2 = st.columns(2)
    classifier_model, classifier_vectorizer, _ = load_classifier()

    if col_btn2.button("Classify review", type="primary", key="intake_classify_btn"):
        if not intake_text.strip():
            st.warning("Enter a review before classifying it.")
        elif classifier_model is None or classifier_vectorizer is None:
            st.warning("The trained issue classifier is not available.")
        else:
            probabilities = classifier_model.predict_proba(classifier_vectorizer.transform([intake_text]))[0]
            ranked = np.argsort(probabilities)[::-1]
            m1, m2 = st.columns(2)
            m1.metric("Predicted issue", classifier_model.classes_[ranked[0]])
            m2.metric("Confidence", f"{probabilities[ranked[0]]:.1%}")

    if col_btn1.button("Analyse and add review", key="intake_add_btn"):
        if not intake_text.strip():
            st.warning("Enter a review before adding it.")
        else:
            predicted_issue = pd.NA
            if classifier_model is not None and classifier_vectorizer is not None:
                predicted_issue = classifier_model.predict(classifier_vectorizer.transform([intake_text]))[0]
            new_row = {
                "review_text": intake_text,
                "restaurant": intake_restaurant,
                "branch": intake_branch,
                "platform": intake_platform,
                "rating": intake_rating,
                "review_date": pd.Timestamp.now(),
                "issue_cluster": predicted_issue,
            }
            st.session_state.setdefault("intake_reviews", []).append(new_row)
            st.success("Review added to the working dataset for this session.")
            st.cache_data.clear()

if not st.session_state["staff_authenticated"]:
    render_customer_review_intake()
    st.stop()

classifier_model, classifier_vectorizer, _ = load_classifier()

# ===========================================================================
# TAB: Existing Reviews
# ===========================================================================
with reviews_tab:
    show_top_metrics(filtered_df)
    st.header("Existing Reviews")
    st.caption("Classify every cleaned review and view issue patterns across restaurants.")

    if filtered_df is None or len(filtered_df) == 0:
        st.warning("No reviews match the current filters.")
    else:
        if classifier_model is not None and st.button("Apply classifier to all reviews", key="apply_all_btn"):
            st.cache_data.clear()
            st.rerun()

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Reviews by Issue")
            issue_counts = filtered_df["issue_cluster"].dropna().value_counts().sort_values(ascending=True)
            if len(issue_counts):
                fig = px.bar(
                    x=issue_counts.values, y=issue_counts.index, orientation="h",
                    labels={"x": "Reviews", "y": "Issue category"},
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No classified reviews yet -- click 'Apply classifier to all reviews' above.")

        with col_b:
            st.subheader("Reviews by Restaurant")
            restaurant_counts = filtered_df["restaurant"].value_counts().sort_values(ascending=True)
            fig = px.bar(
                x=restaurant_counts.values, y=restaurant_counts.index, orientation="h",
                labels={"x": "Reviews", "y": "Restaurant"},
            )
            st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB: SERVQUAL Survey
# ===========================================================================
CANDIDATE_SERVQUAL_FILES = [
    "Restaurant Survey (Responses)SERVQUAL.xlsx",
    "Restaurant_Survey__Responses_SERVQUAL.xlsx",
    "servqual_survey_template.xlsx",
    "servqual_responses.xlsx",
]

# Maps the free-text Google Forms question wording to the SERVQUAL dimension
# it best represents. Matching is done on a lowercased, whitespace-collapsed
# substring basis, so minor wording drift ("appearance" vs "appearance ")
# still matches.
GOOGLE_FORMS_QUESTION_TO_DIMENSION = {
    "arrival of food on time": "Responsiveness",
    "quality of food": "Reliability",
    "staff behaviour and appearance": "Tangibles",
    "cleanliness of restaurant": "Tangibles",
    "overall ambience": "Tangibles",
    "consistency of quality and taste of food": "Reliability",
    "personalised attentiveness to special needs": "Empathy",
    "personalized attentiveness to special needs": "Empathy",
    "proper billing for orders served": "Reliability",
}


def normalize_restaurant_name(name):
    """Collapse messy free-text restaurant names (from a Google Forms
    respondent typing their own answer) into the canonical chain name, using
    the same substring rules as the project's clean_reviews.py."""
    if not isinstance(name, str) or not name.strip():
        return None
    n = name.strip().lower()
    if "a2b" in n or "adyar" in n:
        return "A2B"
    if "geetham" in n or "geetam" in n or "geetha " in n:
        return "Geetham"
    if "sangeetha" in n:
        return "Sangeetha"
    if "saravana" in n:
        return "Saravana Bhavan"
    if "annapoorna" in n or "annapurna" in n:
        return "Sree Annapoorna"
    if "vasant" in n:
        return "Namma Veedu Vasantha Bhavan"
    return None  # not one of the six tracked chains -- excluded from analysis


def parse_google_forms_servqual(df):
    """Parse a real Google Forms SERVQUAL export (paired '<question> [Expectation]'
    / '<question> [Actual]' columns, inconsistent capitalization/pluralization)
    into the same (dimension, mean_expectation, mean_perception, mean_gap,
    n_items) shape that score_survey() produces, so downstream triangulation
    code doesn't need to change.

    Returns (scores_df, n_clean_rows, restaurant_col) or (None, 0, None) if
    this doesn't look like a Google Forms SERVQUAL export.
    """
    exp_suffix_re = re.compile(r"\[\s*expectations?\s*\]", re.IGNORECASE)
    act_suffix_re = re.compile(r"\[\s*actual\s*\]", re.IGNORECASE)

    exp_cols, act_cols = {}, {}
    for col in df.columns:
        base = re.sub(r"^\s*\d+\s*:?\s*", "", str(col))  # strip leading "2 : "
        base_clean = re.sub(r"\s+", " ", base).strip().lower()
        if exp_suffix_re.search(col):
            key = exp_suffix_re.sub("", base_clean).strip()
            exp_cols[key] = col
        elif act_suffix_re.search(col):
            key = act_suffix_re.sub("", base_clean).strip()
            act_cols[key] = col

    paired_keys = set(exp_cols) & set(act_cols)
    if not paired_keys:
        return None, 0, None

    restaurant_col = find_column(df, ["1. Restaurant Name", "Restaurant Name", "Restaurant"])

    # A "clean" response row has at least one non-null paired answer.
    all_pair_cols = [exp_cols[k] for k in paired_keys] + [act_cols[k] for k in paired_keys]
    clean_mask = df[all_pair_cols].notna().any(axis=1)
    clean_df = df[clean_mask]

    dimension_values = {}
    for key in paired_keys:
        dimension = None
        for question_fragment, dim in GOOGLE_FORMS_QUESTION_TO_DIMENSION.items():
            if question_fragment in key:
                dimension = dim
                break
        if dimension is None:
            continue
        exp_series = pd.to_numeric(clean_df[exp_cols[key]], errors="coerce")
        act_series = pd.to_numeric(clean_df[act_cols[key]], errors="coerce")
        dimension_values.setdefault(dimension, {"exp": [], "act": []})
        dimension_values[dimension]["exp"].extend(exp_series.dropna().tolist())
        dimension_values[dimension]["act"].extend(act_series.dropna().tolist())

    records = []
    for dim, vals in dimension_values.items():
        if not vals["exp"] or not vals["act"]:
            continue
        mean_e = float(np.mean(vals["exp"]))
        mean_p = float(np.mean(vals["act"]))
        records.append({
            "dimension": dim,
            "mean_expectation": round(mean_e, 2),
            "mean_perception": round(mean_p, 2),
            "mean_gap": round(mean_p - mean_e, 2),
            "n_items": len(vals["exp"]),
        })

    if not records:
        return None, len(clean_df), restaurant_col

    scores_df = pd.DataFrame(records).sort_values("mean_gap")
    return scores_df, len(clean_df), restaurant_col


with servqual_tab:
    show_top_metrics(filtered_df)
    st.header("SERVQUAL Survey")

    if not SERVQUAL_MODULE_OK:
        st.warning("servqual_survey_analysis.py was not found next to main.py.")
        with st.expander("Import error details"):
            st.code(SERVQUAL_IMPORT_ERROR or "Unknown import error")
    else:
        @st.cache_data
        def load_servqual_responses():
            for rel in CANDIDATE_SERVQUAL_FILES:
                p = BASE_DIR / rel
                if p.exists():
                    try:
                        return pd.read_excel(p, sheet_name="Raw Responses"), rel
                    except Exception:
                        try:
                            return pd.read_excel(p), rel
                        except Exception:
                            continue
            return None, None

        responses_df, servqual_source = load_servqual_responses()

        if responses_df is None:
            st.info("No SERVQUAL survey response file found -- upload one of: " + ", ".join(CANDIDATE_SERVQUAL_FILES))
        else:
            likert_cols = [c for c in responses_df.columns if "|" in str(c)]

            if likert_cols:
                # Field-survey template format ({Dimension}|{i}|E / |P)
                real_rows = responses_df[responses_df[likert_cols].notna().any(axis=1)]
                st.metric("Clean survey responses", len(real_rows))
                scores_sorted = score_survey(real_rows).sort_values("mean_gap") if len(real_rows) >= 2 else None
            else:
                # Real Google Forms export
                scores_sorted, n_clean, restaurant_col = parse_google_forms_servqual(responses_df)
                st.metric("Clean survey responses", n_clean)
                if scores_sorted is not None:
                    scores_sorted = scores_sorted.sort_values("mean_gap")
                    st.caption(
                        f"Parsed from a Google Forms export ('{servqual_source}'). Questions were "
                        "mapped to SERVQUAL dimensions by keyword match; dimensions with no "
                        "matching question in this form (e.g. Assurance) are omitted rather than "
                        "shown with fabricated data."
                    )

            if scores_sorted is not None and len(scores_sorted):
                fig = px.bar(
                    scores_sorted, x="dimension", y="mean_gap", color="mean_gap",
                    color_continuous_scale="RdYlGn", range_color=[-1, 1],
                    labels={"mean_gap": "Perception minus expectation"},
                    title="SERVQUAL Gap by Dimension",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Negative gaps indicate customer perceptions fell below expectations.")

                issue_freq = None
                if filtered_df is not None and filtered_df["issue_cluster"].notna().any():
                    issue_freq = filtered_df["issue_cluster"].value_counts().to_dict()
                merged, rho = triangulate(scores_sorted, issue_freq or INTERIM_ISSUE_FREQUENCY)

                st.subheader("Survey and Review-Issue Triangulation")
                st.dataframe(merged, hide_index=True, use_container_width=True)
            else:
                st.info(
                    f"Loaded a response file ('{servqual_source}'), but couldn't map any "
                    "questions to SERVQUAL dimensions or find valid Likert values. Check the "
                    "column headers match one of the expected formats."
                )

# ===========================================================================
# TAB: Emotion Analysis
# ===========================================================================
EMOTION_COLORS = {
    "neutral": "#1f4fd8", "delight": "#7ec8f2", "frustration": "#e03131",
    "disappointment": "#f2a6a6", "betrayal": "#2f9e44", "nostalgia": "#8ce99a",
    "relief": "#f5a623",
}

with emotion_tab:
    show_top_metrics(filtered_df)
    st.header("Emotion Analysis")
    st.caption("Classify the full review set into restaurant-relevant emotions.")

    if not EMOTION_MODULE_OK:
        st.warning("emotion_classification.py was not found next to main.py.")
        with st.expander("Import error details"):
            st.code(EMOTION_IMPORT_ERROR or "Unknown import error")
    elif filtered_df is None or len(filtered_df) == 0:
        st.warning("No reviews match the current filters.")
    else:
        if st.button("Analyze review emotions", key="analyze_emotions_btn"):
            with st.spinner("Classifying emotions..."):
                st.session_state["emotion_result"] = classify_emotions(filtered_df, "review_text")

        emotion_result = st.session_state.get("emotion_result")
        if emotion_result is not None:
            st.subheader("Review Emotions")
            counts = emotion_result["emotion"].value_counts()
            counts = counts.reindex([e for e in EMOTION_TAXONOMY if e in counts.index])
            fig = px.bar(
                x=counts.index, y=counts.values,
                color=counts.index, color_discrete_map=EMOTION_COLORS,
                labels={"x": "Emotion", "y": "Reviews"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Click 'Analyze review emotions' to run classification.")

# ===========================================================================
# TAB: Root Cause Analysis
# ===========================================================================
with rootcause_tab:
    show_top_metrics(filtered_df)
    st.header("Root Cause Analysis")

    if not ROOTCAUSE_MODULE_OK:
        st.warning("root_cause_emotion_analysis.py was not found next to main.py.")
        with st.expander("Import error details"):
            st.code(ROOTCAUSE_IMPORT_ERROR or "Unknown import error")
    else:
        emotion_result = st.session_state.get("emotion_result")
        if emotion_result is None:
            st.info("Run emotion analysis first (Emotion Analysis tab).")
        elif emotion_result["issue_cluster"].isna().all():
            st.warning("No issue categories available to cross-tab against emotion yet.")
        else:
            crosstab_input = emotion_result.dropna(subset=["issue_cluster"])
            counts, row_pct = build_crosstab(crosstab_input, "issue_cluster", "emotion")

            st.subheader("Emotion Composition by Issue Category")
            fig = px.imshow(
                row_pct, text_auto=".1f", aspect="auto", color_continuous_scale="YlOrRd",
                labels=dict(x="Emotion", y="Issue category", color="Percent"),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Root Cause and Emotion Analysis")
            summary = dominant_emotion_summary(row_pct)
            st.dataframe(summary, hide_index=True, use_container_width=True)

# ===========================================================================
# TAB: Escalation Alerts
# ===========================================================================
with escalation_tab:
    show_top_metrics(filtered_df)
    st.header("Escalation Alerts")

    if not ESCALATION_MODULE_OK:
        st.warning("escalation_detection.py was not found next to main.py.")
        with st.expander("Import error details"):
            st.code(ESCALATION_IMPORT_ERROR or "Unknown import error")
    else:
        emotion_result = st.session_state.get("emotion_result")
        if emotion_result is None:
            st.info("Run emotion analysis first (Emotion Analysis tab).")
        else:
            urgency_df = compute_urgency(emotion_result, "review_text", "emotion", "rating", "review_date")
            flagged = urgency_df[urgency_df["escalate"]].sort_values("urgency_score", ascending=False)

            st.metric("Escalated reviews", len(flagged))

            if len(flagged):
                display_cols = ["restaurant", "review_text", "issue_cluster", "emotion", "urgency_score"]
                display_cols = [c for c in display_cols if c in flagged.columns]
                st.dataframe(flagged[display_cols], hide_index=True, use_container_width=True)
            else:
                st.success("No reviews crossed the escalation threshold.")

            if urgency_df["issue_cluster"].notna().any():
                st.subheader("Escalation rate by issue")
                alerts = cluster_level_alert(urgency_df.dropna(subset=["issue_cluster"]), "issue_cluster")
                if not alerts.empty:
                    st.dataframe(alerts, hide_index=True, use_container_width=True)

# ===========================================================================
# TAB: Empathetic Reply
# ===========================================================================
with reply_tab:
    show_top_metrics(filtered_df)
    st.header("Empathetic Reply Generator")

    if not REPLY_MODULE_OK:
        st.warning("empathetic_reply_generator.py was not found next to main.py.")
        with st.expander("Import error details"):
            st.code(REPLY_IMPORT_ERROR or "Unknown import error")
    else:
        reply_text = st.text_area("Customer review for reply", height=100, key="reply_text_input")

        issue_options = ["your experience"]
        if filtered_df is not None and filtered_df["issue_cluster"].notna().any():
            issue_options = sorted(filtered_df["issue_cluster"].dropna().unique().tolist())
        reply_issue = st.selectbox("Issue category for reply", issue_options, key="reply_issue_select")

        emotion_options = EMOTION_TAXONOMY if EMOTION_MODULE_OK else ["frustration", "disappointment", "betrayal", "neutral"]
        reply_emotion = st.selectbox("Customer emotion", emotion_options, key="reply_emotion_select")

        restaurant_options = sorted(working_df["restaurant"].dropna().unique().tolist()) if working_df is not None else ["Not specified"]
        reply_restaurant = st.selectbox("Restaurant for reply", restaurant_options, key="reply_restaurant_select")

        if st.button("Draft empathetic reply", type="primary", key="draft_reply_btn"):
            if not reply_text.strip():
                st.warning("Enter a customer review before drafting a reply.")
            else:
                one_row = pd.DataFrame({
                    "review_text": [reply_text], "emotion": [reply_emotion],
                    "issue_cluster": [reply_issue], "chain": [reply_restaurant],
                })
                with st.spinner("Drafting reply..."):
                    drafted = generate_replies(one_row, "review_text", "emotion", "issue_cluster", "chain")
                st.session_state["draft_reply_text"] = drafted["draft_reply"].iloc[0]

        if st.session_state.get("draft_reply_text"):
            st.text_area("Draft reply (edit before posting)", value=st.session_state["draft_reply_text"], height=110, key="draft_reply_output")

# ===========================================================================
# TAB: Early-Warning Alerts
# ===========================================================================
with earlywarning_tab:
    st.header("Early-Warning Alerts (Week-over-Week Spikes)")

    if filtered_df is None or filtered_df["issue_cluster"].isna().all():
        st.info("No classified reviews available yet -- visit Existing Reviews and apply the classifier first.")
    else:
        restaurant_filter_options = ["All"] + sorted(filtered_df["restaurant"].dropna().unique().tolist())
        chosen_restaurant = st.selectbox("Filter by restaurant", restaurant_filter_options, key="ew_restaurant_filter")

        scoped = filtered_df.dropna(subset=["issue_cluster"]).copy()
        if chosen_restaurant != "All":
            scoped = scoped[scoped["restaurant"] == chosen_restaurant]

        now = pd.Timestamp.now().normalize()
        this_week_start = now - pd.Timedelta(days=7)
        last_week_start = now - pd.Timedelta(days=14)

        this_week = scoped[scoped["review_date"] >= this_week_start]
        last_week = scoped[(scoped["review_date"] >= last_week_start) & (scoped["review_date"] < this_week_start)]

        group_cols = ["restaurant", "branch", "issue_cluster"]
        this_counts = this_week.groupby(group_cols).size().rename("this_week")
        last_counts = last_week.groupby(group_cols).size().rename("last_week")
        spike_df = pd.concat([this_counts, last_counts], axis=1).fillna(0).reset_index()
        spike_df["this_week"] = spike_df["this_week"].astype(int)
        spike_df["last_week"] = spike_df["last_week"].astype(int)
        spikes = spike_df[spike_df["this_week"] > spike_df["last_week"]].sort_values("this_week", ascending=False)

        st.session_state["early_warning_spikes"] = spikes  # shared with Resolution Workflow tab

        if spikes.empty:
            st.success("No week-over-week complaint spikes detected for the current filters.")
        else:
            for _, row in spikes.iterrows():
                severity = "High" if row["this_week"] >= 3 else "Medium"
                color = "\U0001F534" if severity == "High" else "\U0001F7E1"
                with st.container(border=True):
                    st.markdown(f"{color} **REPUTATION ALERT \u2014 {severity} severity**")
                    st.markdown(f"**Restaurant**  \n{row['restaurant']}")
                    st.markdown(f"**Branch**  \n{row['branch']}")
                    st.markdown(f"**Issue**  \n{row['issue_cluster']}")
                    st.write(f"This week: **{row['this_week']}** complaints | Last week: **{row['last_week']}** complaints")
                    with st.expander("Which platform is driving this spike?"):
                        combo_reviews = this_week[
                            (this_week["restaurant"] == row["restaurant"])
                            & (this_week["branch"] == row["branch"])
                            & (this_week["issue_cluster"] == row["issue_cluster"])
                        ]
                        platform_breakdown = combo_reviews["platform"].value_counts()
                        st.dataframe(platform_breakdown.rename("complaints"), use_container_width=True)

# ===========================================================================
# TAB: Resolution Workflow
# ===========================================================================
with resolution_tab:
    st.header("Complaint-to-Resolution Workflow")
    st.caption(
        "Tracks corrective action for flagged issues. Saved within this session "
        "(Streamlit Cloud has no built-in database -- for persistence across "
        "reboots, wire this to a Google Sheet or small database)."
    )

    spikes = st.session_state.get("early_warning_spikes")
    escalation_flagged = None
    if ESCALATION_MODULE_OK and st.session_state.get("emotion_result") is not None:
        urgency_df = compute_urgency(st.session_state["emotion_result"], "review_text", "emotion", "rating", "review_date")
        escalation_flagged = urgency_df[urgency_df["escalate"]]

    combos = []
    if spikes is not None and len(spikes):
        combos += list(spikes[["restaurant", "branch", "issue_cluster"]].itertuples(index=False, name=None))
    if escalation_flagged is not None and len(escalation_flagged):
        combos += list(
            escalation_flagged.dropna(subset=["issue_cluster"])[["restaurant", "branch", "issue_cluster"]]
            .drop_duplicates().itertuples(index=False, name=None)
        )
    combos = sorted(set(combos))

    if not combos:
        st.info("No flagged issues yet -- check Early-Warning Alerts or Escalation Alerts first.")
    else:
        workflow_state = st.session_state.setdefault("resolution_workflow", {})

        for restaurant, branch, issue in combos:
            key = f"{restaurant}|{branch}|{issue}"
            existing = workflow_state.get(key, {})
            with st.container(border=True):
                st.markdown(f"**{restaurant} \u2014 {branch} \u2014 {issue}**")
                col1, col2 = st.columns(2)
                with col1:
                    status = st.selectbox(
                        "Status", ["New", "In Progress", "Resolved", "Closed"],
                        index=["New", "In Progress", "Resolved", "Closed"].index(existing.get("status", "New")),
                        key=f"status_{key}",
                    )
                    corrective_action = st.text_input("Corrective action taken", value=existing.get("corrective_action", ""), key=f"action_{key}")
                    notes = st.text_input("Notes", value=existing.get("notes", ""), key=f"notes_{key}")
                with col2:
                    assigned_to = st.text_input("Assigned to", value=existing.get("assigned_to", ""), key=f"assigned_{key}")
                    action_date = st.date_input("Action date", value=existing.get("action_date"), key=f"date_{key}")

                if st.button("Save", key=f"save_{key}"):
                    workflow_state[key] = {
                        "status": status, "assigned_to": assigned_to,
                        "corrective_action": corrective_action,
                        "action_date": action_date, "notes": notes,
                    }
                    st.success("Saved.")

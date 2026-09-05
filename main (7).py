"""Unified Streamlit dashboard for restaurant review issue analysis."""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# NOTE: this file is expected to sit at the ROOT of the repo (Streamlit
# Cloud's "Main file path" = "main.py"). If you instead move it into a
# subfolder, change .parent to .parent.parent to match.
BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Restaurant Issue Analysis", layout="wide")
st.title("Restaurant Review Issue Analysis")

# Reputation Management add-on: sits in a "reputation_management/" folder
# alongside main.py. Import is wrapped so the rest of the dashboard still
# works even if that folder hasn't been uploaded yet.
try:
    from reputation_management import (
        MentionMonitor, NLPEngine, RiskPredictor, CompetitiveBenchmark,
        ResponseDrafter, AISearchVisibilityTracker, DigitalFootprintAuditor,
        BiasAuditor, TransparencyLog,
    )
    from reputation_management.trust_ethics import BiasTestCase
    from reputation_management.ai_search_visibility import AISurfaceClient
    from reputation_management.digital_footprint import DiscoverySourceClient, ListingCategory

    REPUTATION_MODULE_AVAILABLE = True
    REPUTATION_IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001 - surfaced to the user in-tab, not crashed
    REPUTATION_MODULE_AVAILABLE = False
    REPUTATION_IMPORT_ERROR = str(exc)

# CX add-on scripts (emotion classification, root-cause x emotion, escalation
# detection, empathetic replies): sit as plain .py files alongside main.py.
try:
    from emotion_classification import classify_emotions, EMOTION_TAXONOMY
    from root_cause_emotion_analysis import build_crosstab, dominant_emotion_summary
    from escalation_detection import compute_urgency, cluster_level_alert
    from empathetic_reply_generator import generate_replies

    CX_MODULES_AVAILABLE = True
    CX_IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001
    CX_MODULES_AVAILABLE = False
    CX_IMPORT_ERROR = str(exc)


@st.cache_data
def load_excel(path):
    file_path = BASE_DIR / path
    return pd.read_excel(file_path) if file_path.exists() else None


@st.cache_data
def load_csv(path):
    file_path = BASE_DIR / path
    return pd.read_csv(file_path) if file_path.exists() else None


@st.cache_data
def load_template_preview():
    template_path = BASE_DIR / "servqual_survey_template.xlsx"
    if not template_path.exists():
        return None, None
    response_form = pd.read_excel(template_path, sheet_name="Response Form", header=None)
    raw_responses = pd.read_excel(template_path, sheet_name="Raw Responses")
    return response_form, raw_responses


@st.cache_resource
def load_classifier():
    file_path = BASE_DIR / "scripts" / "issue_classifier.joblib"
    if not file_path.exists():
        return None, None
    bundle = joblib.load(file_path)
    return bundle["model"], bundle["vectorizer"]


def find_column(dataframe, candidates):
    return next((column for column in candidates if column in dataframe.columns), None)


classified_reviews = load_excel("outputs/reviews_with_issue_classification.xlsx")
anomaly_flags = load_csv("module6_weekly_spike_flags.csv")
priority_spikes = load_csv("module6_priority_ranked_spikes.csv")
cluster_labels = load_excel("outputs/issues/cluster_topics_labeled.xlsx")
servqual_scores = load_csv("servqual_dimension_scores.csv")
servqual_triangulation = load_csv("servqual_nlp_triangulation.csv")
classifier, vectorizer = load_classifier()

classify_tab, reviews_tab, anomaly_tab, priority_tab, clusters_tab, servqual_tab, template_tab, reputation_tab, emotion_tab, about_tab = st.tabs(
    [
        "Issue Classification",
        "Existing Reviews",
        "Anomaly Trends",
        "Priority Ranking",
        "Cluster Analysis",
        "SERVQUAL Survey",
        "Survey Template",
        "Reputation Management",
        "Emotion & Escalation",
        "About This Project",
    ]
)

with classify_tab:
    st.header("Classify a New Review")
    if classifier is None or vectorizer is None:
        st.warning("The trained issue classifier is not available.")
    else:
        restaurants = ["Not specified"]
        if classified_reviews is not None:
            restaurant_column = find_column(classified_reviews, ["restaurant", "Restaurant"])
            if restaurant_column:
                restaurants = sorted(classified_reviews[restaurant_column].dropna().unique())
        st.selectbox("Restaurant", restaurants, key="new_review_restaurant")
        review_text = st.text_area("Review text", height=160)
        if st.button("Classify review", type="primary"):
            if review_text.strip():
                probabilities = classifier.predict_proba(vectorizer.transform([review_text]))[0]
                ranked_indices = np.argsort(probabilities)[::-1]
                st.metric("Predicted issue", classifier.classes_[ranked_indices[0]])
                st.metric("Confidence", f"{probabilities[ranked_indices[0]]:.1%}")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Issue category": classifier.classes_[ranked_indices[:3]],
                            "Probability": probabilities[ranked_indices[:3]],
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.warning("Enter a review before classifying it.")

with reviews_tab:
    st.header("Existing Review Classifications")
    if classified_reviews is None:
        st.warning("Classified reviews are not available.")
    else:
        issue_column = find_column(
            classified_reviews,
            ["predicted_issue_category", "issue_category"],
        )
        restaurant_column = find_column(classified_reviews, ["restaurant", "Restaurant"])
        confidence_column = find_column(
            classified_reviews,
            ["confidence", "prediction_confidence"],
        )
        if issue_column is None:
            st.warning("The classified reviews file has no issue category column.")
        else:
            restaurant_filter, issue_filter = st.columns(2)
            restaurant_options = ["All restaurants"]
            if restaurant_column:
                restaurant_options += sorted(
                    classified_reviews[restaurant_column].dropna().unique()
                )
            selected_restaurant = restaurant_filter.selectbox(
                "Restaurant",
                restaurant_options,
                key="reviews_restaurant_filter",
            )
            selected_issue = issue_filter.selectbox(
                "Issue category",
                ["All issues"] + sorted(classified_reviews[issue_column].dropna().unique()),
                key="reviews_issue_filter",
            )

            filtered_reviews = classified_reviews.copy()
            if restaurant_column and selected_restaurant != "All restaurants":
                filtered_reviews = filtered_reviews[
                    filtered_reviews[restaurant_column] == selected_restaurant
                ]
            if selected_issue != "All issues":
                filtered_reviews = filtered_reviews[
                    filtered_reviews[issue_column] == selected_issue
                ]

            metric_one, metric_two, metric_three = st.columns(3)
            metric_one.metric("Reviews", len(filtered_reviews))
            metric_two.metric("Issue categories", filtered_reviews[issue_column].nunique())
            average_confidence = "N/A"
            if confidence_column and not filtered_reviews.empty:
                average_confidence = f"{filtered_reviews[confidence_column].mean():.1%}"
            metric_three.metric("Average confidence", average_confidence)

            category_counts = filtered_reviews[issue_column].value_counts().reset_index()
            category_counts.columns = ["Issue category", "Reviews"]
            st.plotly_chart(
                px.bar(category_counts, x="Reviews", y="Issue category", orientation="h"),
                use_container_width=True,
            )
            if restaurant_column and not filtered_reviews.empty:
                restaurant_counts = pd.crosstab(
                    filtered_reviews[restaurant_column], filtered_reviews[issue_column]
                ).reset_index()
                issue_counts = [
                    column for column in restaurant_counts.columns if column != restaurant_column
                ]
                st.plotly_chart(
                    px.bar(
                        restaurant_counts,
                        x=restaurant_column,
                        y=issue_counts,
                        barmode="stack",
                        labels={"value": "Reviews", "variable": "Issue category"},
                    ),
                    use_container_width=True,
                )

with anomaly_tab:
    st.header("Anomaly and Trend Detection")
    if anomaly_flags is None:
        st.warning("Anomaly trend results are not available.")
    else:
        anomaly_flags = anomaly_flags.copy()
        anomaly_flags["week"] = pd.to_datetime(anomaly_flags["week"])
        confirmed_spikes = anomaly_flags[anomaly_flags["both_flag"]]
        rule_metric, forest_metric, confirmed_metric = st.columns(3)
        rule_metric.metric("Z-score anomalies", int(anomaly_flags["rule_flag"].sum()))
        forest_metric.metric("Isolation Forest anomalies", int(anomaly_flags["iforest_flag"].sum()))
        confirmed_metric.metric("Confirmed spikes", len(confirmed_spikes))
        categories = sorted(anomaly_flags["issue_category"].dropna().unique())
        selected_categories = st.multiselect("Issue categories", categories, default=categories)
        filtered_anomalies = anomaly_flags[
            anomaly_flags["issue_category"].isin(selected_categories)
        ]
        st.plotly_chart(
            px.line(
                filtered_anomalies,
                x="week",
                y="review_count",
                color="issue_category",
                labels={"week": "Week", "review_count": "Review count"},
            ),
            use_container_width=True,
        )
        visible_spikes = confirmed_spikes[
            confirmed_spikes["issue_category"].isin(selected_categories)
        ]
        if not visible_spikes.empty:
            st.plotly_chart(
                px.scatter(
                    visible_spikes,
                    x="week",
                    y="z_score",
                    size="review_count",
                    color="issue_category",
                    hover_data=["rolling_mean"],
                ),
                use_container_width=True,
            )
        st.dataframe(
            filtered_anomalies.sort_values("z_score", ascending=False),
            hide_index=True,
            use_container_width=True,
        )

with priority_tab:
    st.header("Priority-Ranked Spikes")
    if priority_spikes is None:
        st.warning("Priority ranking results are not available.")
    else:
        priority_spikes = priority_spikes.copy()
        priority_spikes["week"] = pd.to_datetime(priority_spikes["week"])
        categories = sorted(priority_spikes["issue_category"].dropna().unique())
        selected_categories = st.multiselect(
            "Issue categories",
            categories,
            default=categories,
            key="priority_issue_categories",
        )
        ranked_spikes = priority_spikes[
            priority_spikes["issue_category"].isin(selected_categories)
        ].sort_values("predicted_priority_score", ascending=False)

        top_score = ranked_spikes["predicted_priority_score"].max()
        metric_one, metric_two, metric_three = st.columns(3)
        metric_one.metric("Ranked spike-weeks", len(ranked_spikes))
        metric_two.metric("Highest priority score", f"{top_score:.3f}" if pd.notna(top_score) else "N/A")
        metric_three.metric("Issue categories", ranked_spikes["issue_category"].nunique())

        st.subheader("Learned priority by spike magnitude")
        st.plotly_chart(
            px.scatter(
                ranked_spikes,
                x="magnitude_z",
                y="predicted_priority_score",
                size="review_count",
                color="issue_category",
                hover_data=["week", "pct_above_rolling_mean"],
                labels={
                    "magnitude_z": "Z-score magnitude",
                    "predicted_priority_score": "Learned priority score",
                    "review_count": "Review count",
                },
            ),
            use_container_width=True,
        )

        st.subheader("Highest-priority spikes")
        display_columns = [
            "issue_category",
            "week",
            "review_count",
            "magnitude_z",
            "pct_above_rolling_mean",
            "predicted_priority_score",
        ]
        st.dataframe(
            ranked_spikes[display_columns].reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
        )

        shap_path = BASE_DIR / "module6_shap_summary.png"
        if shap_path.exists():
            st.subheader("XGBoost feature importance (SHAP)")
            st.image(shap_path, use_container_width=True)

with clusters_tab:
    st.header("Cluster Analysis")
    if cluster_labels is None:
        st.warning("Cluster labels are not available.")
    else:
        st.dataframe(cluster_labels, hide_index=True, use_container_width=True)

    cluster_viz_path = BASE_DIR / "cluster_visualization.png"
    if cluster_viz_path.exists():
        st.subheader("2D Cluster Projection (UMAP)")
        st.image(
            str(cluster_viz_path),
            caption="Customer complaint clusters projected to 2D via UMAP",
            use_container_width=True,
        )

with servqual_tab:
    st.header("SERVQUAL Survey Analysis")
    if servqual_scores is None or servqual_triangulation is None:
        st.warning("SERVQUAL results are not available. Run the survey analysis script first.")
    else:
        servqual_scores = servqual_scores.sort_values("mean_gap")
        worst_dimension = servqual_scores.iloc[0]
        total_nlp_frequency = servqual_triangulation["matched_nlp_frequency"].sum()

        metric_one, metric_two, metric_three = st.columns(3)
        metric_one.metric("Largest service gap", worst_dimension["dimension"])
        metric_two.metric("Gap score", f"{worst_dimension['mean_gap']:.2f}")
        metric_three.metric("Mapped NLP complaints", f"{total_nlp_frequency:,}")

        st.subheader("Expected vs. perceived service")
        expectation_perception = servqual_scores.melt(
            id_vars="dimension",
            value_vars=["mean_expectation", "mean_perception"],
            var_name="Measure",
            value_name="Score",
        )
        expectation_perception["Measure"] = expectation_perception["Measure"].map(
            {
                "mean_expectation": "Expectation",
                "mean_perception": "Perception",
            }
        )
        st.plotly_chart(
            px.bar(
                expectation_perception,
                x="dimension",
                y="Score",
                color="Measure",
                barmode="group",
                range_y=[0, 7],
                labels={"dimension": "SERVQUAL dimension"},
            ),
            use_container_width=True,
        )

        st.subheader("Service gaps by dimension")
        st.plotly_chart(
            px.bar(
                servqual_scores,
                x="mean_gap",
                y="dimension",
                orientation="h",
                color="mean_gap",
                color_continuous_scale="RdYlGn",
                labels={"mean_gap": "Perception minus expectation"},
            ),
            use_container_width=True,
        )

        st.subheader("SERVQUAL and review-issue triangulation")
        st.dataframe(
            servqual_triangulation.sort_values("mean_gap"),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Negative gap values indicate service perceptions fell below customer expectations. "
            "The current analysis uses the script's synthetic survey sample."
        )

with template_tab:
    st.header("SERVQUAL Survey Template")
    template_path = BASE_DIR / "servqual_survey_template.xlsx"
    if not template_path.exists():
        st.warning("The survey template is not available. Run the template generator first.")
    else:
        metric_one, metric_two, metric_three = st.columns(3)
        metric_one.metric("SERVQUAL dimensions", 5)
        metric_two.metric("Survey statements", 20)
        metric_three.metric("Prepared respondent rows", 15)
        st.download_button(
            "Download survey template (.xlsx)",
            data=template_path.read_bytes(),
            file_name="servqual_survey_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Worksheet": ["Instructions", "Response Form", "Raw Responses"],
                    "Purpose": [
                        "Field-collection guidance",
                        "Printable single-respondent questionnaire",
                        "Analysis-ready ratings for up to 15 respondents",
                    ],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        response_form_preview, raw_responses_preview = load_template_preview()
        with st.expander("Preview response form"):
            st.dataframe(
                response_form_preview.fillna(""),
                hide_index=True,
                use_container_width=True,
            )
        with st.expander("Preview raw response entry sheet"):
            st.dataframe(
                raw_responses_preview.head(15),
                hide_index=True,
                use_container_width=True,
            )

with reputation_tab:
    st.header("Reputation Management")

    if not REPUTATION_MODULE_AVAILABLE:
        st.warning(
            "The 'reputation_management' add-on package was not found next to "
            "main.py. Upload the 'reputation_management' folder (from the "
            "reputation_management_addon.zip) to the repo root to enable this tab."
        )
        with st.expander("Import error details"):
            st.code(REPUTATION_IMPORT_ERROR or "Unknown import error")
    else:
        st.caption(
            "Cross-platform mention monitoring, sentiment velocity, predictive "
            "risk flags, AI-assisted response drafting, AI-search visibility, "
            "digital footprint auditing, and bias/transparency auditing — built "
            "on top of the same review data used elsewhere in this dashboard."
        )

        @st.cache_data
        def _load_mention_source_texts():
            """Pull real review text + dates where available, else fall back
            to a small synthetic sample so this tab still works standalone."""
            candidates = [classified_reviews, cluster_labels]
            for candidate in candidates:
                if candidate is not None and len(candidate) > 0:
                    text_col = find_column(candidate, ["review_text", "review", "Review"])
                    date_col = find_column(candidate, ["review_date", "date", "Date"])
                    if text_col:
                        sample = candidate[[text_col]].dropna().head(200).copy()
                        sample = sample.rename(columns={text_col: "text"})
                        if date_col and date_col in candidate.columns:
                            sample["timestamp"] = pd.to_datetime(
                                candidate.loc[sample.index, date_col], errors="coerce"
                            )
                        return sample.reset_index(drop=True), False
            sample_reviews = [
                "The wait time was ridiculous, waited 40 minutes for a simple meal!!",
                "Absolutely loved the food, amazing flavors and quick service.",
                "Yeah great service... waited an hour and food was cold. Wow!!!",
                "Staff was rude and negligent, wouldn't recommend at all.",
                "Kitchen looked dirty and there was an unhygienic smell near the counter.",
                "Overcharged on the bill, had to ask for a refund, very annoying.",
                "no cap this place slaps, best value for money fr fr",
                "Food quality was tasteless and stale, quite disappointed.",
                "Great portions, fair price, will come again!",
                "Rude staff again, second time this happened, unacceptable.",
            ]
            return pd.DataFrame({"text": sample_reviews}), True

        mention_source_df, using_demo_data = _load_mention_source_texts()
        if using_demo_data:
            st.info(
                "No review text found in already-loaded data, so this tab is "
                "running on a small built-in demo sample. It will automatically "
                "switch to your real reviews once available."
            )

        @st.cache_resource
        def _build_monitor(_source_df):
            from datetime import datetime, timedelta
            import random

            from reputation_management.monitoring import SourceClient

            now = datetime.utcnow()
            has_dates = "timestamp" in _source_df.columns

            def _fetch():
                rows = []
                for i, row in _source_df.reset_index(drop=True).iterrows():
                    ts = row["timestamp"] if has_dates and pd.notna(row.get("timestamp")) else (
                        now - timedelta(minutes=random.randint(0, 6000))
                    )
                    rows.append({
                        "id": f"m{i}",
                        "source": "review_feed",
                        "text": str(row["text"]),
                        "author": None,
                        "timestamp": ts,
                    })
                return rows

            monitor = MentionMonitor(velocity_window_hours=24 * 30)
            monitor.register_source(SourceClient("review_feed", _fetch))
            mentions = monitor.poll_all_sources()
            return monitor, mentions

        monitor, mentions = _build_monitor(mention_source_df)

        (
            monitoring_subtab, nlp_subtab, risk_subtab, response_subtab,
            visibility_subtab, footprint_subtab, trust_subtab,
        ) = st.tabs([
            "Monitoring & Velocity", "NLP Analyzer", "Risk Flags",
            "Response Drafting", "AI Search Visibility",
            "Digital Footprint Audit", "Trust & Transparency",
        ])

        # -- Monitoring & Sentiment Velocity ---------------------------------
        with monitoring_subtab:
            st.subheader("Sentiment Velocity")
            st.caption(
                "How fast negative sentiment is accelerating, bucketed by time "
                "window — the early-warning signal, not just current sentiment level."
            )
            velocity = monitor.sentiment_velocity(bucket_minutes=60 * 24 * 7)
            if not velocity:
                st.info("Not enough mention volume to compute sentiment velocity yet.")
            else:
                velocity_df = pd.DataFrame(velocity)
                fig = px.line(
                    velocity_df, x="bucket_start", y="negative_share",
                    markers=True, title="Negative Share Over Time",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(velocity_df, hide_index=True, use_container_width=True)

                anomalies = monitor.detect_anomalies(bucket_minutes=60 * 24 * 7)
                st.subheader(f"Anomaly Alerts ({len(anomalies)})")
                if anomalies:
                    st.dataframe(pd.DataFrame(anomalies), hide_index=True, use_container_width=True)
                else:
                    st.success("No statistically significant sentiment spikes detected.")

        # -- NLP Deep-Dive -----------------------------------------------------
        with nlp_subtab:
            st.subheader("Analyze a Review's Tone")
            st.caption("Goes beyond positive/negative: sarcasm, dominant emotion, and cultural-context flags.")
            sample_text = st.text_area(
                "Review text", value=mention_source_df["text"].iloc[0] if len(mention_source_df) else "",
                height=100,
            )
            if st.button("Analyze tone", key="analyze_tone_btn"):
                result = NLPEngine().analyze(sample_text)
                c1, c2, c3 = st.columns(3)
                c1.metric("Polarity", f"{result.polarity_label} ({result.polarity_score})")
                c2.metric("Sarcasm flagged", "Yes" if result.sarcasm_flag else "No")
                c3.metric("Dominant emotion", result.dominant_emotion)
                if result.needs_human_review:
                    st.warning("Flagged for human review (sarcasm, cultural context, or ambiguous score).")
                st.json(result.emotions)

        # -- Predictive Risk Flags ----------------------------------------------
        with risk_subtab:
            st.subheader("Predictive Risk Flags")
            st.caption(
                "Combines complaint-topic volume, negativity share, and sentiment "
                "velocity into an early-warning risk score per topic."
            )
            predictor = RiskPredictor()
            flags = predictor.score(mentions, velocity_by_bucket=monitor.sentiment_velocity(bucket_minutes=60 * 24 * 7))
            if flags:
                flags_df = pd.DataFrame([f.__dict__ for f in flags])
                st.dataframe(flags_df, hide_index=True, use_container_width=True)
            else:
                st.success("No elevated or critical risk topics detected in the current data.")

        # -- AI-Assisted Response Drafting ---------------------------------------
        with response_subtab:
            st.subheader("Draft a Response")
            st.caption("Tone-matched draft reply with a mandatory human-approval step before it's considered final.")
            if not mentions:
                st.info("No mentions available to draft a response for.")
            else:
                options = {f"{m.id}: {m.text[:60]}...": m for m in mentions[:30]}
                choice = st.selectbox("Pick a mention", list(options.keys()))
                selected_mention = options[choice]

                if "response_drafter" not in st.session_state:
                    st.session_state.response_drafter = ResponseDrafter()
                drafter = st.session_state.response_drafter

                if st.button("Generate draft", key="draft_btn"):
                    draft = drafter.draft(selected_mention)
                    st.session_state[f"draft_{selected_mention.id}"] = draft

                draft = st.session_state.get(f"draft_{selected_mention.id}")
                if draft:
                    st.text_area("Draft reply", value=draft.draft_text, height=100, key=f"draft_text_{selected_mention.id}")
                    st.caption(f"Tone: {draft.tone} | Status: {draft.status}")
                    col_a, col_b = st.columns(2)
                    if col_a.button("Approve", key=f"approve_{selected_mention.id}"):
                        drafter.approve(selected_mention.id, reviewer="dashboard_user")
                        st.success("Approved.")
                    if col_b.button("Reject", key=f"reject_{selected_mention.id}"):
                        drafter.reject(selected_mention.id, reviewer="dashboard_user")
                        st.error("Rejected.")

        # -- AI Search Visibility ------------------------------------------------
        with visibility_subtab:
            st.subheader("AI Search Visibility")
            st.caption("Tracks whether/how your brand appears in AI Overviews or assistant answers for key prompts.")
            brand_name = st.text_input("Brand name", value="Our Restaurant")
            tracked_prompt = st.text_input("Prompt to check", value="best vegetarian restaurant nearby")
            if st.button("Run visibility check", key="visibility_btn"):
                def _fake_ai_answer(prompt: str) -> str:
                    return f"For vegetarian dining, {brand_name} is a popular, well-reviewed option nearby."

                tracker = AISearchVisibilityTracker(brand_name, tracked_prompts=[tracked_prompt])
                tracker.register_surface(AISurfaceClient("ai_overview", _fake_ai_answer))
                checks = tracker.run_checks()
                st.dataframe(pd.DataFrame([c.__dict__ for c in checks]), hide_index=True, use_container_width=True)
                st.caption(
                    "Uses a placeholder AI-answer function — replace `AISurfaceClient`'s "
                    "query function with a real AI-search API call for production use."
                )

        # -- Digital Footprint Audit ----------------------------------------------
        with footprint_subtab:
            st.subheader("Digital Footprint Audit")
            st.caption("Self-service audit + takedown-request workflow for personal listings (requires explicit consent).")
            subject_name = st.text_input("Name to audit", value="")
            consent = st.checkbox("I confirm this is my own name and I consent to this audit.")
            if st.button("Run audit", key="footprint_btn", disabled=not (subject_name and consent)):
                def _fake_broker_search(name: str):
                    return [{
                        "url": "https://databroker.example/profile/123",
                        "snippet": f"{name} - phone: 98xxxxxxx0, address on file",
                        "is_outdated": True,
                    }]

                auditor = DigitalFootprintAuditor(subject_name=subject_name, consent_confirmed=True)
                auditor.register_source(
                    DiscoverySourceClient("ExampleDataBroker", ListingCategory.DATA_BROKER, _fake_broker_search)
                )
                listings = auditor.run_audit()
                st.dataframe(pd.DataFrame(auditor.takedown_dashboard()), hide_index=True, use_container_width=True)
                if listings:
                    auditor.request_takedown(listings[0].id)
                    st.success(f"Takedown requested for listing {listings[0].id}.")
                st.caption(
                    "Uses a placeholder data-broker search — replace `DiscoverySourceClient`'s "
                    "search function with a real data-broker/opt-out API for production use."
                )

        # -- Trust, Bias & Transparency --------------------------------------------
        with trust_subtab:
            st.subheader("Bias Audit")
            st.caption("Checks whether the sentiment model scores equivalent statements differently based on dialect/register.")
            bias_cases = [
                BiasTestCase(
                    group_a_text="The service was straight-up bad, not gonna lie.",
                    group_a_label="AAVE_slang",
                    group_b_text="The service was quite poor, to be honest.",
                    group_b_label="standard_english",
                ),
                BiasTestCase(
                    group_a_text="Food was great fr fr no cap.",
                    group_a_label="AAVE_slang",
                    group_b_text="The food was genuinely excellent.",
                    group_b_label="standard_english",
                ),
            ]
            bias_auditor = BiasAuditor()
            report = bias_auditor.audit(bias_cases)
            st.dataframe(pd.DataFrame(BiasAuditor.summarize(report)), hide_index=True, use_container_width=True)

            st.subheader("Transparency Log")
            st.caption("Audit trail of any automated score changes, with a human-readable reason and contributing evidence.")
            if "transparency_log" not in st.session_state:
                st.session_state.transparency_log = TransparencyLog()
                st.session_state.transparency_log.record(
                    entity="Overall Reputation Score", metric="reputation_score",
                    new_value=72.5, old_value=78.0,
                    reason="Spike in wait-time-related negative mentions (sentiment velocity accelerating)",
                    contributing_mention_ids=[m.id for m in mentions[:3]],
                )
            st.dataframe(
                pd.DataFrame(st.session_state.transparency_log.dashboard_feed()),
                hide_index=True, use_container_width=True,
            )


with emotion_tab:
    st.header("Emotion, Root-Cause & Escalation Analysis")

    if not CX_MODULES_AVAILABLE:
        st.warning(
            "The CX add-on scripts (emotion_classification.py, "
            "root_cause_emotion_analysis.py, escalation_detection.py, "
            "empathetic_reply_generator.py, cx_common.py) were not found "
            "next to main.py. Upload all five files to the repo root to "
            "enable this tab."
        )
        with st.expander("Import error details"):
            st.code(CX_IMPORT_ERROR or "Unknown import error")
    else:
        st.caption(
            "Goes beyond positive/negative sentiment: classifies each review's "
            "dominant emotion, cross-tabs emotion against issue clusters, flags "
            "reviews needing urgent attention, and drafts empathetic manager "
            "replies. Uses the Groq API if GROQ_API_KEY is set in secrets, "
            "otherwise falls back to keyword/template heuristics automatically."
        )

        # Bridge Streamlit secrets -> environment variable, since cx_common.py
        # reads GROQ_API_KEY via os.environ (works with or without secrets set).
        try:
            if "GROQ_API_KEY" in st.secrets and not os.environ.get("GROQ_API_KEY"):
                os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass  # no secrets.toml configured -- fine, heuristic fallback is used

        cx_source_df = None
        for candidate in (classified_reviews, cluster_labels):
            if candidate is not None and len(candidate) > 0:
                cx_source_df = candidate
                break

        if cx_source_df is None:
            st.info(
                "No review dataset loaded yet (looks for "
                "outputs/reviews_with_issue_classification.xlsx). Using a "
                "small built-in demo sample instead."
            )
            cx_source_df = pd.DataFrame({
                "review_text": [
                    "The wait time was ridiculous, waited 40 minutes for a simple meal!!",
                    "Absolutely loved the food, amazing flavors and quick service.",
                    "Portion was tiny for the price, felt cheated honestly.",
                    "Staff was rude and negligent, wouldn't recommend at all.",
                    "Kitchen looked dirty and there was an unhygienic smell near the counter.",
                    "Overcharged on the bill, had to ask for a refund, very annoying.",
                    "Used to be so much better a few years ago, quality has dropped.",
                    "Food quality was tasteless and stale, quite disappointed.",
                    "Great portions, fair price, will come again!",
                    "Surprised how quickly they fixed the mix-up with our order.",
                ],
                "rating": [1, 5, 2, 1, 1, 2, 3, 2, 5, 4],
            })

        cx_text_col = find_column(cx_source_df, ["review_text", "review", "Review"]) or "review_text"
        cx_rating_col = find_column(cx_source_df, ["rating", "Rating"])
        cx_date_col = find_column(cx_source_df, ["review_date", "date", "Date"])
        cx_chain_col = find_column(cx_source_df, ["restaurant", "chain", "Restaurant"])
        cx_issue_col = find_column(
            cx_source_df, ["predicted_issue_category", "issue_cluster", "issue_category"]
        )

        emo_subtab, root_cause_subtab, escalation_subtab, reply_subtab = st.tabs(
            ["Emotion Classification", "Root Cause x Emotion", "Escalation Detection", "Empathetic Reply Drafts"]
        )

        # -- Emotion Classification -----------------------------------------------
        with emo_subtab:
            st.subheader("Emotion-Layer Classification")
            st.caption(
                "Classifies each review's dominant emotion (delight, "
                "disappointment, frustration, betrayal, relief, nostalgia, "
                "neutral) -- more actionable than plain positive/negative."
            )
            max_n = max(min(300, len(cx_source_df)), 1)
            default_n = min(50, max_n)
            sample_n = st.slider(
                "Number of reviews to classify (capped for demo speed)",
                min_value=min(5, max_n), max_value=max_n, value=default_n,
            )
            if st.button("Run emotion classification", key="run_emotion_btn"):
                subset = cx_source_df.head(sample_n).copy()
                with st.spinner("Classifying emotions..."):
                    st.session_state["cx_emotion_result"] = classify_emotions(subset, cx_text_col)

            emotion_result = st.session_state.get("cx_emotion_result")
            if emotion_result is not None:
                counts = emotion_result["emotion"].value_counts().reset_index()
                counts.columns = ["emotion", "count"]
                fig = px.bar(counts, x="emotion", y="count", title="Emotion Distribution")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(
                    emotion_result[[cx_text_col, "emotion", "emotion_conf"]].head(50),
                    hide_index=True, use_container_width=True,
                )
            else:
                st.info("Click 'Run emotion classification' to see results.")

        # -- Root Cause x Emotion ---------------------------------------------------
        with root_cause_subtab:
            st.subheader("Root Cause x Emotion Cross-Tab")
            st.caption(
                "Which emotion dominates each business issue? e.g. frustration "
                "clustering around wait times, betrayal around portion/value complaints."
            )
            emotion_result = st.session_state.get("cx_emotion_result")
            if emotion_result is None:
                st.info("Run emotion classification first (previous sub-tab).")
            elif not cx_issue_col or cx_issue_col not in emotion_result.columns:
                st.warning(
                    "No issue-category column found in the loaded data to cross-tab "
                    "against emotion (expected 'predicted_issue_category' or 'issue_cluster')."
                )
            else:
                counts, row_pct = build_crosstab(emotion_result, cx_issue_col, "emotion")
                fig = px.imshow(
                    row_pct, text_auto=".0f", aspect="auto", color_continuous_scale="YlOrRd",
                    labels=dict(x="Emotion", y="Issue Cluster", color="% of cluster"),
                    title="Emotion Composition per Issue Cluster (row %)",
                )
                st.plotly_chart(fig, use_container_width=True)
                summary = dominant_emotion_summary(row_pct)
                st.dataframe(summary, hide_index=True, use_container_width=True)

        # -- Escalation Detection -----------------------------------------------------
        with escalation_subtab:
            st.subheader("Escalation / Urgency Detection")
            st.caption(
                "Transparent, tunable urgency score = emotion weight + severity-keyword "
                "weight + rating weight + recency weight. Flags reviews needing immediate "
                "attention instead of waiting for the next pipeline re-run."
            )
            emotion_result = st.session_state.get("cx_emotion_result")
            if emotion_result is None:
                st.info("Run emotion classification first (first sub-tab).")
            else:
                urgency_df = compute_urgency(
                    emotion_result, cx_text_col, "emotion",
                    cx_rating_col or "rating", cx_date_col,
                )
                flagged = urgency_df[urgency_df["escalate"]].sort_values("urgency_score", ascending=False)

                c1, c2 = st.columns(2)
                c1.metric("Reviews analyzed", len(urgency_df))
                c2.metric(
                    "Escalated", len(flagged),
                    f"{len(flagged) / max(len(urgency_df), 1) * 100:.1f}%",
                )

                display_cols = [cx_text_col, "urgency_score"]
                if cx_rating_col and cx_rating_col in flagged.columns:
                    display_cols.append(cx_rating_col)
                if flagged.empty:
                    st.success("No reviews crossed the escalation threshold in this sample.")
                else:
                    st.dataframe(flagged[display_cols], hide_index=True, use_container_width=True)

                if cx_issue_col and cx_issue_col in urgency_df.columns:
                    alerts = cluster_level_alert(urgency_df, cx_issue_col)
                    if not alerts.empty:
                        st.subheader("Escalation Rate by Issue Cluster")
                        st.dataframe(alerts, hide_index=True, use_container_width=True)

        # -- Empathetic Reply Drafts -----------------------------------------------------
        with reply_subtab:
            st.subheader("Draft an Empathetic Reply")
            st.caption(
                "Acknowledges the SPECIFIC emotion + issue instead of a generic "
                "'sorry for the inconvenience' template. Always a draft for human "
                "review/edit before posting -- never auto-posted."
            )
            draft_review_text = st.text_area(
                "Review text",
                value=cx_source_df[cx_text_col].iloc[0] if len(cx_source_df) else "",
                height=90, key="reply_review_text",
            )
            col_a, col_b, col_c = st.columns(3)
            draft_emotion = col_a.selectbox("Detected emotion", EMOTION_TAXONOMY, key="reply_emotion")
            draft_issue = col_b.text_input("Issue category", value="your experience", key="reply_issue")
            draft_chain = col_c.text_input("Restaurant chain (optional)", value="", key="reply_chain")

            if st.button("Draft reply", key="draft_reply_btn"):
                one_row = pd.DataFrame({
                    "review_text": [draft_review_text],
                    "emotion": [draft_emotion],
                    "issue_cluster": [draft_issue],
                    "chain": [draft_chain],
                })
                with st.spinner("Drafting reply..."):
                    drafted = generate_replies(one_row, "review_text", "emotion", "issue_cluster", "chain")
                st.session_state["cx_draft_reply"] = drafted["draft_reply"].iloc[0]

            draft_text = st.session_state.get("cx_draft_reply")
            if draft_text:
                st.text_area("Draft reply (edit before posting)", value=draft_text, height=100, key="draft_reply_output")


with about_tab:
    st.header("About This Project")
    readme_path = BASE_DIR / "README.md"
    if readme_path.exists():
        st.markdown(readme_path.read_text(encoding="utf-8"))
    else:
        st.markdown(
            """
This dashboard implements **Module 6 — Reputation Early-Warning & Resolution**
for the Customer Experience Analytics project (BITS Pilani MBA Dissertation,
2024MB22535), covering:

- **Real-time issue classification** — a supervised classifier (Random
  Forest / Logistic Regression / Gradient Boosting) trained on HDBSCAN +
  Groq-labeled reviews, tagging new incoming reviews into issue categories
  instantly.
- **Anomaly & trend detection** — a transparent z-score rule cross-checked
  with an Isolation Forest model to flag genuine complaint-volume spikes.
- **Priority ranking** — an XGBoost regressor (explained via SHAP) that
  learns which historical spikes actually preceded a rating drop, replacing
  a manually-weighted severity formula.
- **SERVQUAL triangulation** — a mini 10–15 respondent field survey
  compared against the NLP cluster frequency ranking, corroborating the
  automatically-discovered issues with direct customer feedback.

*(Add a `README.md` at the repo root to customize this tab.)*
            """
        )

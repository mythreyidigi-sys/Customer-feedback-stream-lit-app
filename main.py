"""Unified Streamlit dashboard for restaurant review issue analysis."""
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from scripts.ai_search_visibility import AISearchVisibilityTracker, AISurfaceClient
from scripts.digital_footprint import DigitalFootprintAuditor, DiscoverySourceClient, ListingCategory
from scripts.monitoring import MentionMonitor
from scripts.nlp_engine import NLPEngine
from scripts.prediction import CompetitiveBenchmark, RiskPredictor
from scripts.response import ResponseDrafter
from scripts.trust_ethics import BiasAuditor, BiasTestCase, TransparencyLog

# NOTE: this file is expected to sit at the ROOT of the repo (Streamlit
# Cloud's "Main file path" = "main.py"). If you instead move it into a
# subfolder, change .parent to .parent.parent to match.
BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Restaurant Issue Analysis", layout="wide")
st.title("Restaurant Review Issue Analysis")


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


@st.cache_data
def load_findings_showcase():
    review_path = BASE_DIR / "outputs" / "cleaned_reviews.xlsx"
    if not review_path.exists():
        return None

    reviews = pd.read_excel(review_path).dropna(subset=["review"])
    reference_time = datetime.utcnow()
    monitor = MentionMonitor(nlp_engine=NLPEngine(), velocity_window_hours=24)
    mentions = monitor.ingest_batch([
        {
            "id": f"a2b-review-{index + 1}",
            "source": "google_reviews",
            "text": review,
            "author": None,
            "timestamp": reference_time - timedelta(
                hours=(len(reviews) - 1 - index) * 24 / max(len(reviews), 1)
            ),
        }
        for index, review in enumerate(reviews["review"].astype(str))
    ])
    velocity = monitor.sentiment_velocity(bucket_minutes=120)
    risk_flags = RiskPredictor().score(mentions, velocity)
    response = ResponseDrafter().draft(mentions[0]) if mentions else None

    visibility = AISearchVisibilityTracker(
        "A2B", ["best vegetarian restaurant in Chennai", "A2B customer feedback"]
    )
    visibility.register_surface(AISurfaceClient(
        "demo_ai",
        lambda prompt: f"A2B is popular and well-reviewed for South Indian food. Prompt: {prompt}",
    ))
    visibility.run_checks()

    footprint = DigitalFootprintAuditor("consenting demo subject", consent_confirmed=True)
    footprint.register_source(DiscoverySourceClient(
        "demo search index",
        ListingCategory.SEARCH_RESULT,
        lambda query: [{
            "url": "https://example.test/review",
            "snippet": f"Public review summary for {query}: restaurant feedback",
        }],
    ))
    footprint.run_audit()

    benchmark = CompetitiveBenchmark()
    benchmark.add_brand_mentions("A2B", mentions)
    bias_report = BiasAuditor(nlp_engine=NLPEngine()).audit([
        BiasTestCase("The food was good.", "standard", "The food was tasty.", "regional")
    ])
    transparency = TransparencyLog()
    transparency.record(
        "A2B",
        "topic_risk_score",
        risk_flags[0].risk_score if risk_flags else 0.0,
        "Derived from imported Google review mentions",
        contributing_mention_ids=[mention.id for mention in mentions[:3]],
    )
    return {
        "reviews": len(reviews),
        "first_review": mentions[0].nlp if mentions else None,
        "velocity_buckets": len(velocity),
        "anomalies": len(monitor.detect_anomalies(bucket_minutes=120)),
        "risk_flags": risk_flags,
        "response": response,
        "visibility_rate": visibility.visibility_rate(),
        "visibility_summary": visibility.trend_summary(),
        "footprint": footprint.takedown_dashboard(),
        "benchmark": benchmark.compare(["food_quality", "wait_time", "hygiene"]),
        "bias_audit": BiasAuditor.summarize(bias_report),
        "transparency_events": len(transparency.dashboard_feed()),
    }


def find_column(dataframe, candidates):
    return next((column for column in candidates if column in dataframe.columns), None)


classified_reviews = load_excel("outputs/reviews_with_issue_classification.xlsx")
anomaly_flags = load_csv("module6_weekly_spike_flags.csv")
priority_spikes = load_csv("module6_priority_ranked_spikes.csv")
cluster_labels = load_excel("outputs/issues/cluster_topics_labeled.xlsx")
servqual_scores = load_csv("servqual_dimension_scores.csv")
servqual_triangulation = load_csv("servqual_nlp_triangulation.csv")
classifier, vectorizer = load_classifier()

classify_tab, reviews_tab, anomaly_tab, priority_tab, clusters_tab, servqual_tab, template_tab, findings_tab, about_tab = st.tabs(
    [
        "Issue Classification",
        "Existing Reviews",
        "Anomaly Trends",
        "Priority Ranking",
        "Cluster Analysis",
        "SERVQUAL Survey",
        "Survey Template",
        "Findings Showcase",
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

with findings_tab:
    st.header("Cross-Module Findings Showcase")
    st.caption("Full-dataset analysis using all cleaned reviews from clean_reviews.py and local, consent-safe demo clients.")
    findings = load_findings_showcase()
    if findings is None:
        st.warning("The A2B Google Reviews file is not available.")
    else:
        first_review = findings["first_review"]
        risk_flags = findings["risk_flags"]
        metric_one, metric_two, metric_three, metric_four = st.columns(4)
        metric_one.metric("Reviews analyzed", findings["reviews"])
        metric_two.metric("Food-quality risk", risk_flags[0].risk_score if risk_flags else "N/A")
        metric_three.metric("AI visibility", f"{findings['visibility_rate']:.0%}")
        metric_four.metric("Bias review flags", findings["bias_audit"][0]["flagged_pairs"])

        st.subheader("What the review stream found")
        finding_rows = [{
            "Finding": "NLP sentiment",
            "Result": f"First review: {first_review.polarity_label} ({first_review.polarity_score:.3f}), emotion: {first_review.dominant_emotion}",
            "Evidence": "VADER polarity plus emotion tagging",
        }, {
            "Finding": "Monitoring",
            "Result": f"{findings['velocity_buckets']} time buckets, {findings['anomalies']} anomalies",
            "Evidence": "Sentiment velocity and anomaly detection",
        }, {
            "Finding": "Early warning",
            "Result": risk_flags[0].rationale if risk_flags else "No threshold-crossing topic",
            "Evidence": "Topic volume, negative share, and velocity",
        }, {
            "Finding": "Response workflow",
            "Result": f"{findings['response'].tone} tone, status: {findings['response'].status.value}",
            "Evidence": "Draft is pending human approval",
        }]
        st.dataframe(pd.DataFrame(finding_rows), hide_index=True, use_container_width=True)

        left_column, right_column = st.columns(2)
        with left_column:
            st.subheader("Benchmark and visibility")
            st.dataframe(pd.DataFrame(findings["benchmark"]), hide_index=True, use_container_width=True)
            st.dataframe(pd.DataFrame(findings["visibility_summary"]), hide_index=True, use_container_width=True)
        with right_column:
            st.subheader("Governance evidence")
            st.dataframe(pd.DataFrame(findings["bias_audit"]), hide_index=True, use_container_width=True)
            st.dataframe(pd.DataFrame(findings["footprint"]), hide_index=True, use_container_width=True)
            st.metric("Transparency events logged", findings["transparency_events"])

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

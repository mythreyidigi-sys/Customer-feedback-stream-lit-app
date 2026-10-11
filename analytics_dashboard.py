"""Power BI-style analytics page for the Empathy Engine (rendered as screen 7 in main.py).

Reads the pipeline output outputs/reviews_with_issue_classification.xlsx (so it always reflects the latest
scrape + pipeline run), scores sentiment with VADER and draws the dashboard charts.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

CLASSIFIED = Path("outputs") / "reviews_with_issue_classification.xlsx"
CHAIN_NAMES = {
    "a2b": "A2B (Adyar Ananda Bhavan)", "adyar": "A2B (Adyar Ananda Bhavan)",
    "vasant": "Vasantha Bhavan (Namma Veedu)", "namma": "Vasantha Bhavan (Namma Veedu)",
    "saravana": "Saravana Bhavan", "sangeetha": "Sangeetha", "geetham": "Geetham", "annapoorna": "Sree Annapoorna",
    "annapurna": "Sree Annapoorna", "sree": "Sree Annapoorna", "sri ": "Sree Annapoorna",
}
POSITIVE_CUTOFF, NEGATIVE_CUTOFF = 0.5, 0.0
OVERPRICED = re.compile(r"over\s?priced|too (?:expensive|costly|pricey)|very (?:expensive|costly|pricey)|"
                        r"costly|expensive|not worth|waste of money|rip[- ]?off|high price|highly priced|"
                        r"price(?:s)? (?:is|are) (?:high|steep)", re.I)
GOOD_VALUE = re.compile(r"value for money|worth (?:the|every|it)|reasonabl[ey]|affordable|good price|"
                        r"fair price|budget|cheap|pocket[- ]friendly|economical|inexpensive", re.I)
PRICE_WORDS = re.compile(r"\b(price|prices|pricing|cost|rate|rates|bill|charged?|rs\.?|₹|rupees)\b", re.I)
STOPWORDS = {"the", "and", "was", "for", "with", "this", "that", "are", "not", "but", "very", "had", "have",
             "you", "they", "our", "its", "were", "from", "has", "all", "can", "one", "also", "there", "their",
             "which", "will", "would", "been", "just", "get", "got", "when", "than", "then", "too", "out", "only",
             "here", "what", "more", "some", "any", "into", "about", "after", "your", "restaurant", "food"}


def chain_of(name: str) -> str:
    low = str(name).lower()
    for key, label in CHAIN_NAMES.items():
        if key in low:
            return label
    return str(name)


def source_of(path: str) -> str:
    low = str(path).lower().replace("\\", "/")
    if "zomato" in low:
        return "Zomato"
    if "trip" in low:
        return "TripAdvisor"
    return "Google"


def price_perception(text: str) -> str:
    if OVERPRICED.search(text):
        return "Feels overpriced"
    if GOOD_VALUE.search(text):
        return "Good value"
    if PRICE_WORDS.search(text):
        return "Mentions price"
    return "No price mention"


@st.cache_data(show_spinner="Scoring sentiment…")
def load_dashboard_data(path: str, modified: float) -> pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    df = pd.read_excel(path)
    analyser = SentimentIntensityAnalyzer()
    text = df["review"].fillna(df.get("review_text", "")).astype(str)
    df["restaurant"] = df["restaurant"].map(chain_of)
    df["source"] = df["source_file"].map(source_of)
    df["sentiment_score"] = text.map(lambda t: analyser.polarity_scores(t)["compound"])
    df["segment"] = pd.cut(df["sentiment_score"], [-2, NEGATIVE_CUTOFF, POSITIVE_CUTOFF - 1e-9, 2],
                           labels=["Detractor", "Neutral", "Promoter"])
    df["segment"] = df["segment"].astype(str)
    df["price_perception"] = text.map(price_perception)
    df["text_for_cloud"] = text
    return df


def kpi_row(items: list[tuple[str, str]]) -> None:
    for col, (label, value) in zip(st.columns(len(items)), items):
        with col.container(border=True):
            st.metric(label, value)


def word_cloud(texts: pd.Series):
    from wordcloud import WordCloud

    words = " ".join(texts.str.lower())
    return WordCloud(width=900, height=420, background_color="white", stopwords=STOPWORDS,
                     colormap="tab10", collocations=False).generate(words).to_image()


def render(root: Path) -> None:
    path = root / CLASSIFIED
    if not path.is_file():
        st.warning(f"Pipeline output not found: {path}. Run automation/run_pipeline.py first.")
        return
    df = load_dashboard_data(str(path), path.stat().st_mtime)

    with st.expander("Filters", expanded=False):
        f1, f2 = st.columns(2)
        chosen_rest = f1.multiselect("Restaurants", sorted(df["restaurant"].unique()))
        chosen_src = f2.multiselect("Sources", sorted(df["source"].unique()))
    if chosen_rest:
        df = df[df["restaurant"].isin(chosen_rest)]
    if chosen_src:
        df = df[df["source"].isin(chosen_src)]
    if df.empty:
        st.info("No reviews match these filters.")
        return

    st.caption("Sentiment = VADER compound score (−1 to +1) on each review. Promoter ≥ 0.5, Detractor ≤ 0, "
               "otherwise Neutral. Price perception is keyword-based.")
    tab_overview, tab_issues, tab_segments = st.tabs(["Overview", "Issues & sentiment", "Segments & price"])

    by_rest = df.groupby("restaurant").agg(reviews=("review", "size"), avg_sentiment=("sentiment_score", "mean"),
                                           branches=("branch", "nunique")).reset_index()

    with tab_overview:
        kpi_row([("Count of review", f"{len(df):,}"), ("Count of branch", f"{df['branch'].nunique():,}"),
                 ("Count of restaurant", str(df["restaurant"].nunique())),
                 ("Count of source", str(df["source"].nunique()))])
        c1, c2, c3 = st.columns([1.2, 1.4, 1.2])
        c1.markdown("**Count of review by restaurant**")
        c1.plotly_chart(px.bar(by_rest.sort_values("reviews", ascending=False), x="restaurant", y="reviews")
                        .update_layout(margin=dict(t=10, b=10), xaxis_title=None), width="stretch")
        c2.markdown("**Count of review by restaurant and source**")
        rs = df.groupby(["restaurant", "source"]).size().reset_index(name="reviews")
        c2.plotly_chart(px.bar(rs, y="restaurant", x="reviews", color="source", orientation="h")
                        .update_layout(margin=dict(t=10, b=10), yaxis_title=None,
                                       yaxis={"categoryorder": "total ascending"}), width="stretch")
        c3.markdown("**Avg sentiment by restaurant**")
        c3.plotly_chart(px.bar(by_rest.sort_values("avg_sentiment"), y="restaurant", x="avg_sentiment",
                               orientation="h").update_layout(margin=dict(t=10, b=10), yaxis_title=None),
                        width="stretch")
        d1, d2, d3 = st.columns([1.2, 1.4, 1.2])
        d1.markdown("**Count of review by source**")
        src = df["source"].value_counts().rename_axis("source").reset_index(name="reviews")
        d1.plotly_chart(px.bar(src, y="source", x="reviews", orientation="h")
                        .update_layout(margin=dict(t=10, b=10), yaxis_title=None), width="stretch")
        d2.markdown("**Reviews per restaurant**")
        table = by_rest.set_index("restaurant")[["reviews"]].sort_values("reviews", ascending=False)
        table.loc["Total"] = table["reviews"].sum()
        d2.dataframe(table, width="stretch")
        d3.markdown("**Most-used words**")
        try:
            d3.image(word_cloud(df["text_for_cloud"]), width="stretch")
        except ImportError:
            d3.info("Install the 'wordcloud' package to show the word cloud.")

    with tab_issues:
        top = (df["predicted_issue_category"].value_counts().head(1).index[0]
               if "predicted_issue_category" in df else "n/a")
        leader = by_rest.loc[by_rest["avg_sentiment"].idxmax()]
        kpi_row([("Overall avg sentiment score", f"{df['sentiment_score'].mean():.2f}"),
                 ("Sentiment leader", f"{leader['restaurant']} ({leader['avg_sentiment']:.3f})"),
                 ("Most frequent issue", top)])
        e1, e2 = st.columns(2)
        e1.markdown("**Sum of frequency by issue**")
        issues = df["predicted_issue_category"].value_counts().rename_axis("issue").reset_index(name="frequency")
        e1.plotly_chart(px.bar(issues, y="issue", x="frequency", orientation="h")
                        .update_layout(margin=dict(t=10, b=10), yaxis={"categoryorder": "total ascending"}),
                        width="stretch")
        e2.markdown("**Review count, avg sentiment and branch count by restaurant**")
        e2.plotly_chart(px.scatter(by_rest, x="reviews", y="avg_sentiment", size="branches", color="restaurant",
                                   size_max=30).update_layout(margin=dict(t=10, b=10)), width="stretch")
        g1, g2 = st.columns(2)
        g1.markdown("**Avg sentiment by restaurant and source**")
        pivot = df.pivot_table(index="restaurant", columns="source", values="sentiment_score", aggfunc="mean")
        pivot["Total"] = df.groupby("restaurant")["sentiment_score"].mean()
        pivot.loc["Total"] = [df[df["source"] == s]["sentiment_score"].mean() for s in pivot.columns[:-1]] + \
                             [df["sentiment_score"].mean()]
        g1.dataframe(pivot.style.format("{:.3f}").background_gradient(cmap="Blues", axis=None),
                     width="stretch")
        g2.markdown("**Issue counts by restaurant**")
        g2.dataframe(pd.crosstab(df["restaurant"], df["predicted_issue_category"], margins=True,
                                 margins_name="Total"), width="stretch")

    with tab_segments:
        seg_counts = df["segment"].value_counts()
        promoters, detractors = seg_counts.get("Promoter", 0), seg_counts.get("Detractor", 0)
        total = len(df)
        by_price = df.groupby("price_perception")["sentiment_score"].mean()
        gap = by_price.get("Good value", float("nan")) - by_price.get("Feels overpriced", float("nan"))
        kpi_row([("Net sentiment score (promoter % − detractor %)", f"{(promoters - detractors) / total * 100:.1f}"),
                 ("Promoter / detractor ratio", f"{promoters / detractors:.2f}" if detractors else "n/a"),
                 ("Value perception gap (good value − overpriced)", f"{gap:.2f}" if pd.notna(gap) else "n/a"),
                 ("% feels overpriced", f"{(df['price_perception'] == 'Feels overpriced').mean() * 100:.1f}%")])
        h1, h2 = st.columns([1, 1.6])
        h1.markdown("**Customer segments**")
        h1.plotly_chart(px.pie(seg_counts.rename_axis("segment").reset_index(name="reviews"), names="segment",
                               values="reviews", hole=0.55, color="segment",
                               color_discrete_map={"Detractor": "#D85A30", "Neutral": "#B4B2A9",
                                                   "Promoter": "#1D9E75"}).update_layout(margin=dict(t=10, b=10)),
                        width="stretch")
        h2.markdown("**Avg sentiment by price perception and restaurant**")
        pp = df.groupby(["price_perception", "restaurant"])["sentiment_score"].mean().reset_index()
        h2.plotly_chart(px.bar(pp, x="price_perception", y="sentiment_score", color="restaurant", barmode="group",
                               category_orders={"price_perception": ["Good value", "Mentions price",
                                                                      "No price mention", "Feels overpriced"]})
                        .update_layout(margin=dict(t=10, b=10)), width="stretch")
        st.markdown("**Avg sentiment by source and restaurant**")
        sr = df.groupby(["source", "restaurant"])["sentiment_score"].mean().reset_index()
        st.plotly_chart(px.bar(sr, x="source", y="sentiment_score", color="restaurant", barmode="group")
                        .update_layout(margin=dict(t=10, b=10)), width="stretch")

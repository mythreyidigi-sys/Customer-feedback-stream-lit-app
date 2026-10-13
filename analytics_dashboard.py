"""Power BI-style analytics page for the Empathy Engine (screen 1 in main.py).

Works on the reviews already filtered by the sidebar, scores sentiment with VADER and draws the charts.
"""
from __future__ import annotations

import re

import pandas as pd
import plotly.express as px
import streamlit as st

CHAIN_NAMES = {
    "a2b": "A2B (Adyar Ananda Bhavan)", "adyar": "A2B (Adyar Ananda Bhavan)",
    "vasant": "Vasantha Bhavan (Namma Veedu)", "namma": "Vasantha Bhavan (Namma Veedu)",
    "saravana": "Saravana Bhavan", "sangeetha": "Sangeetha", "geetham": "Geetham", "annapoorna": "Sree Annapoorna",
    "annapurna": "Sree Annapoorna", "sree": "Sree Annapoorna", "sri ": "Sree Annapoorna",
}
POSITIVE_CUTOFF, NEGATIVE_CUTOFF = 0.5, 0.0
NOISE_LABELS = {"noise", "unclassified", "uncategorized", "uncategorised", "outlier", "-1", "other", "",
                "brand reputation & customer satisfaction"}
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


@st.cache_data(show_spinner="Scoring sentiment?")
def _vader_scores(texts: tuple[str, ...]) -> list[float]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyser = SentimentIntensityAnalyzer()
    return [analyser.polarity_scores(t)["compound"] for t in texts]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Adapt the sidebar-filtered app reviews to the columns the dashboard charts use."""
    df = df.copy()
    text = df["text"].fillna("").astype(str)
    df["review"] = text
    df["predicted_issue_category"] = df["issue"].where(
        ~df["issue"].astype("string").str.strip().str.lower().isin(NOISE_LABELS))
    df["sentiment_score"] = _vader_scores(tuple(text))
    df["segment"] = pd.cut(df["sentiment_score"], [-2, NEGATIVE_CUTOFF, POSITIVE_CUTOFF - 1e-9, 2],
                           labels=["Detractor", "Neutral", "Promoter"]).astype(str)
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


def top_issues_table(ctx: dict) -> None:
    """Top issues for the selected restaurant, with mentions and change vs the previous window."""
    an, cur, prev = ctx["an"], ctx["cur"], ctx["prev"]
    st.markdown(f"**Top issues ? {ctx['restaurant']}** ? change vs previous {ctx['days']} days")
    ti = an.top_issues(cur, prev)
    ti = ti[~ti["issue"].astype("string").str.strip().str.lower().isin(NOISE_LABELS)]
    if ti.empty:
        st.info("No issues in this date range.")
        return
    ti["trend"] = ti["change_pct"].apply(
        lambda v: "new" if pd.isna(v) else ("▲ " if v > 0 else "▼ " if v < 0 else "= ") + f"{abs(v):.0f}%")
    st.dataframe(ti[["issue", "mentions", "previous", "trend"]], hide_index=True, width="stretch",
                 column_config={"mentions": st.column_config.ProgressColumn(
                     "Mentions", format="%d", min_value=0, max_value=int(ti["mentions"].max() or 1)),
                     "previous": "Previous", "trend": "Change", "issue": "Issue"})


PIPELINE_STEPS = [("All reviews", "Selected chains + rivals"), ("Embed together", "Same Sentence-Transformer"),
                  ("HDBSCAN once", "Shared clusters"), ("Groq labels", "Name each cluster")]
CHAIN_COLOURS = ["#1D9E75", "#D85A30", "#7F77DD", "#E0A526", "#3B8BD4", "#B4B2A9"]


def cluster_share_chart(df: pd.DataFrame, max_clusters: int = 12) -> None:
    """Share of each ML cluster (HDBSCAN + LLM label) contributed by each chain: crosstab normalised by cluster."""
    cols = st.columns(len(PIPELINE_STEPS))
    for col, (title, sub) in zip(cols, PIPELINE_STEPS):
        col.markdown(f"<div style='border:1px solid #ddd;border-radius:12px;padding:10px;text-align:center;"
                     f"background:#f7f5f0'><b>{title}</b><br><small>{sub}</small></div>", unsafe_allow_html=True)
    data = df[~df["issue"].astype("string").str.strip().str.lower().isin(NOISE_LABELS)]
    if data["restaurant"].nunique() < 2 or data.empty:
        st.info("Select at least one competitor in 'Compare with' to see how each cluster splits across chains.")
        return
    top = data["issue"].value_counts().head(max_clusters).index
    data = data[data["issue"].isin(top)]
    share = pd.crosstab(data["issue"], data["restaurant"], normalize="index") * 100
    counts = data["issue"].value_counts()
    share = share.loc[counts.index[::-1]]
    fig = px.bar(share.reset_index().melt(id_vars="issue", var_name="restaurant", value_name="share"),
                 y="issue", x="share", color="restaurant", orientation="h", text="share",
                 color_discrete_sequence=CHAIN_COLOURS, category_orders={"issue": list(share.index)})
    fig.update_traces(texttemplate="%{text:.0f}%", textposition="inside")
    fig.update_layout(barmode="stack", xaxis=dict(range=[0, 100], ticksuffix="%", title=None), yaxis_title=None,
                      legend_title=None, legend=dict(orientation="h", y=-0.15), margin=dict(t=10, b=10),
                      height=max(320, 55 * len(share) + 80))
    st.markdown("**Share of each cluster by chain**")
    st.plotly_chart(fig, width="stretch")
    st.caption("Share per cluster = pd.crosstab(cluster, chain, normalize='index'). Clusters are the shared "
               "HDBSCAN groups already stored on each review, so every chain is measured on the same labels.")


def reputation_tab(df: pd.DataFrame, an, issues_ctx: dict) -> None:
    """Reputation score per restaurant/branch/month and the issue clusters behind it."""
    clustered = df[~df["issue"].astype("string").str.strip().str.lower().isin(NOISE_LABELS)]
    score = an.reputation_score(df)
    neg = (df["sentiment"] == "negative").mean() * 100
    top = clustered.loc[clustered["issue"] != an.POSITIVE_ISSUE, "issue"].value_counts()
    kpi_row([("Reputation score (0-100)", f"{score:.1f}"), ("Reviews", f"{len(df):,}"),
             ("Negative share", f"{neg:.1f}%"), ("Issue clusters", str(clustered["issue"].nunique())),
             ("Biggest complaint cluster", top.index[0] if len(top) else "n/a")])
    st.caption("Reputation score = 60% average rating (1-5 rescaled) + 40% net sentiment "
               "(positive share − negative share). Clusters come from Sentence-Transformer + HDBSCAN, labelled by LLM.")
    c1, c2 = st.columns(2)
    scores = pd.DataFrame([{"restaurant": r, "score": an.reputation_score(d), "reviews": len(d)}
                           for r, d in df.groupby("restaurant")]).sort_values("score")
    c1.markdown("**Reputation score by restaurant**")
    c1.plotly_chart(px.bar(scores, y="restaurant", x="score", orientation="h", text="score", range_x=[0, 100])
                    .update_layout(margin=dict(t=10, b=10), yaxis_title=None), width="stretch")
    trend = an.monthly_trend(df)
    c2.markdown("**Reputation score by month**")
    if len(trend):
        c2.plotly_chart(px.line(trend, x="month", y="score", markers=True)
                        .update_layout(margin=dict(t=10, b=10)), width="stretch")
    cl = clustered.groupby("issue").agg(reviews=("id", "size"), negative_pct=("sentiment", lambda s: (s == "negative").mean() * 100)).reset_index()
    d1, d2 = st.columns(2)
    d1.markdown("**Reviews per cluster**")
    d1.plotly_chart(px.bar(cl, y="issue", x="reviews", orientation="h", color="negative_pct",
                           color_continuous_scale="RdYlGn_r")
                    .update_layout(margin=dict(t=10, b=10), yaxis={"categoryorder": "total ascending"},
                                   yaxis_title=None), width="stretch")
    with d2.container(border=True):
        top_issues_table(issues_ctx)
    st.markdown("**Reputation by restaurant and branch**")
    rows = [{"restaurant": r, "branch": b, "reviews": len(d), "score": an.reputation_score(d),
             "negative_pct": round((d["sentiment"] == "negative").mean() * 100, 1)}
            for (r, b), d in df.groupby(["restaurant", "branch"])]
    st.dataframe(pd.DataFrame(rows).sort_values("score"), width="stretch", hide_index=True)


DISH_PATTERNS = {
    "Ghee roast": r"ghee\s?roast", "Filter coffee": r"filter\s?coffee|\bcoffee\b", "Pongal": r"pongal",
    "Meals": r"\bmeals?\b|thali", "Dosa": r"\bdosai?\b|masala dosa", "Idli": r"\bidli|idly", "Vada": r"\bvada\b|\bvadai\b",
    "Sambar": r"sambh?ar", "Biryani": r"biryani|briyani", "Parotta": r"parott?ha|porotta", "Sweets": r"sweets?|mysore pak|halwa|jangiri|laddu",
    "Chaat": r"chaa?t|pani puri|pav bhaji", "Tea": r"\btea\b|\bchai\b", "Rice items": r"variety rice|curd rice|lemon rice|pulao",
}
DISH_SAMPLE_PER_CHAIN = 150
DISH_STEPS = [("Reviews", "Chosen chains"), ("Groq: dish + score", "Per-dish sentiment"),
              ("Mean per dish × chain", "Gap = A − B")]


def _dish_hits(text: str) -> list[str]:
    return [dish for dish, pat in DISH_PATTERNS.items() if re.search(pat, text, re.I)]


@st.cache_data(show_spinner="Scoring dishes with Groq…")
def score_dishes(items: tuple[tuple[str, str, float], ...]) -> tuple[pd.DataFrame, str]:
    """(chain, review, vader) -> rows of chain, dish, score (-1..1). Groq extracts dish + score; VADER fallback."""
    import json

    try:
        from engine.config import GROQ_API_KEY, GROQ_MODEL
        from engine.replies import _groq
    except ImportError:
        GROQ_API_KEY, GROQ_MODEL, _groq = "", "", None
    rows, used_groq = [], False
    for chain, text, vader in items:
        dishes = _dish_hits(text)
        if not dishes:
            continue
        scores = {}
        if GROQ_API_KEY and _groq:
            prompt = (f"Review: \"{text[:600]}\"\nFor each of these dishes mentioned in the review: {dishes}, give "
                      f"the customer's sentiment about that dish from -1 (very negative) to 1 (very positive). "
                      f"Reply only with JSON like {{\"dish\": 0.5}}.")
            out = _groq(prompt, max_tokens=80, temperature=0) or ""
            try:
                parsed = json.loads(out[out.index("{"):out.rindex("}") + 1])
                scores = {d: float(v) for d, v in parsed.items() if d in dishes and -1 <= float(v) <= 1}
                used_groq = used_groq or bool(scores)
            except (ValueError, TypeError):
                scores = {}
        for dish in dishes:
            rows.append({"chain": chain, "dish": dish, "score": scores.get(dish, vader)})
    engine = (f"Groq LLM · {GROQ_MODEL} (dish extraction + sentiment scoring, VADER fallback per review)"
              if used_groq else "VADER sentiment on dish keyword matches (set GROQ_API_KEY for the Groq LLM)")
    return pd.DataFrame(rows, columns=["chain", "dish", "score"]), engine


def dish_comparison_section(df: pd.DataFrame, own: str) -> None:
    """Mean dish sentiment for two chains side by side, with the gap."""
    st.markdown("### Dish-level sentiment: head to head")
    cols = st.columns(len(DISH_STEPS))
    for col, (title, sub) in zip(cols, DISH_STEPS):
        col.markdown(f"<div style='border:1px solid #ddd;border-radius:12px;padding:10px;text-align:center;"
                     f"background:#f7f5f0'><b>{title}</b><br><small>{sub}</small></div>", unsafe_allow_html=True)
    chains = sorted(df["restaurant"].unique())
    if len(chains) < 2:
        st.info("Select at least one competitor in 'Compare with' (sidebar) to compare dishes.")
        return
    a_col, b_col = st.columns(2)
    a = a_col.selectbox("Chain A", chains, index=chains.index(own) if own in chains else 0, key="dish_a")
    b = b_col.selectbox("Chain B", [c for c in chains if c != a], key="dish_b")
    sub = pd.concat([d.sort_values("review_date", ascending=False).head(DISH_SAMPLE_PER_CHAIN)
                     for _, d in df[df["restaurant"].isin([a, b])].groupby("restaurant")])
    sub = sub[sub["review"].map(lambda t: bool(_dish_hits(t)))]
    scored, engine = score_dishes(tuple(zip(sub["restaurant"], sub["review"], sub["sentiment_score"])))
    means = scored.pivot_table(index="dish", columns="chain", values="score", aggfunc="mean")
    counts = scored.pivot_table(index="dish", columns="chain", values="score", aggfunc="size")
    if a not in means or b not in means:
        st.info("Not enough dish mentions for both chains in this period.")
        return
    means = means.dropna(subset=[a, b])
    means = means[(counts.loc[means.index, a] >= 2) & (counts.loc[means.index, b] >= 2)]
    if means.empty:
        st.info("No dish has at least 2 mentions for both chains in this period.")
        return
    means["gap"] = means[a] - means[b]
    means = means.sort_values("gap")
    long = means.reset_index().melt(id_vars=["dish", "gap"], value_vars=[a, b], var_name="chain", value_name="score")
    fig = px.bar(long, y="dish", x="score", color="chain", barmode="group", orientation="h", text="score",
                 color_discrete_sequence=["#1D9E75", "#D85A30"], category_orders={"dish": list(means.index)})
    fig.update_traces(texttemplate="%{text:+.2f}", textposition="outside")
    for dish, gap in means["gap"].items():
        fig.add_annotation(xref="paper", yref="y", x=1.02, y=dish, text=f"<b>{gap:+.2f}</b>", showarrow=False,
                           font=dict(color="#1D9E75" if gap > 0 else "#D85A30"))
    fig.add_annotation(xref="paper", yref="paper", x=1.02, y=1.04, text="Gap (A − B)", showarrow=False)
    fig.update_layout(margin=dict(t=30, b=10, r=90), legend_title=None, yaxis_title=None,
                      xaxis_title="Mean dish sentiment (−1 to +1)", height=max(320, 90 * len(means) + 100))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"ML model used: **{engine}**. Gap = {a} − {b}; dishes need ≥ 2 mentions per chain. "
               f"Latest {DISH_SAMPLE_PER_CHAIN} reviews per chain are scored.")


def competitor_tab(df: pd.DataFrame, an, issues_ctx: dict) -> None:
    """Head-to-head view of the selected restaurant against the chains chosen in 'Compare with'."""
    own = issues_ctx["restaurant"]
    if df["restaurant"].nunique() < 2:
        st.info("Select at least one competitor in 'Compare with' (sidebar) to unlock the cluster comparison.")
        dish_comparison_section(df, own)
        return
    rows = [{"restaurant": r, "reputation": an.reputation_score(d)} for r, d in df.groupby("restaurant")]
    comp = pd.DataFrame(rows).sort_values("reputation", ascending=False)
    rivals = comp[comp["restaurant"] != own]
    own_row = comp[comp["restaurant"] == own]
    if len(own_row) and len(rivals):
        score = own_row["reputation"].iloc[0]
        kpi_row([(f"{own} reputation score", f"{score:.1f}"),
                 ("Rank among compared chains", f"{int((comp['reputation'] > score).sum()) + 1} of {len(comp)}"),
                 ("Gap vs rival average", f"{score - rivals['reputation'].mean():+.1f}"),
                 ("Best rival", f"{rivals.iloc[0]['restaurant']} ({rivals.iloc[0]['reputation']:.1f})")])
    cluster_share_chart(df)
    st.markdown("**Reviews per cluster and chain**")
    clustered = df[~df["issue"].astype("string").str.strip().str.lower().isin(NOISE_LABELS)]
    st.dataframe(pd.crosstab(clustered["issue"], clustered["restaurant"], margins=True, margins_name="Total"),
                 width="stretch")
    dish_comparison_section(df, own)


def render(df: pd.DataFrame, an, issues_ctx: dict) -> None:
    """df is the sidebar-filtered review set from main.py (restaurant, branches, sources, date range)."""
    if df.empty:
        st.info("No reviews match the sidebar filters. Widen the date range or change filters in the sidebar.")
        return
    df = prepare(df)

    st.caption("Sentiment = VADER compound score (−1 to +1) on each review. Promoter ≥ 0.5, Detractor ≤ 0, "
               "otherwise Neutral. Price perception is keyword-based.")
    tab_overview, tab_rep, tab_comp, tab_issues, tab_segments = st.tabs(
        ["Overview", "Reputation scoring and clustering", "Competitor analysis", "Issues & sentiment",
         "Segments & price"])

    by_rest = df.groupby("restaurant").agg(reviews=("review", "size"), avg_sentiment=("sentiment_score", "mean"),
                                           branches=("branch", "nunique")).reset_index()

    with tab_rep:
        reputation_tab(df, an, issues_ctx)

    with tab_comp:
        competitor_tab(df, an, issues_ctx)

    with tab_overview:
        kpi_row([("Count of review", f"{len(df):,}"), ("Count of branch", f"{df['branch'].nunique():,}"),
                 ("Count of restaurant", str(df["restaurant"].nunique())),
                 ("Count of source", str(df["source"].nunique()))])
        c1, c2, c3 = st.columns([1.2, 1.2, 1.4])
        c1.markdown("**Count of review by restaurant**")
        c1.plotly_chart(px.bar(by_rest.sort_values("reviews", ascending=False), x="restaurant", y="reviews")
                        .update_layout(margin=dict(t=10, b=10), xaxis_title=None), width="stretch")
        c2.markdown("**Count of review by source**")
        src = df["source"].value_counts().rename_axis("source").reset_index(name="reviews")
        c2.plotly_chart(px.bar(src, y="source", x="reviews", orientation="h")
                        .update_layout(margin=dict(t=10, b=10), yaxis_title=None), width="stretch")
        c3.markdown("**Most-used words**")
        try:
            c3.image(word_cloud(df["text_for_cloud"]), width="stretch")
        except ImportError:
            c3.info("Install the 'wordcloud' package to show the word cloud.")

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

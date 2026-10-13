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
WIN_LOSS_STEPS = [("All reviews", "Every chain"), ("Name filter", "Regex on rivals"),
                  ("Groq classify", "Win / loss / neutral"), ("Net win rate", "Tracked monthly")]


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


RIVAL_PATTERNS = {
    "A2B": r"\ba2b\b|adyar ananda", "Sangeetha": r"sangeetha|sangeeta", "Geetham": r"\bgeetham\b|\bgeetam\b",
    "Saravana Bhavan": r"saravana", "Vasantha Bhavan": r"vasantha|namma veedu",
    "Sree Annapoorna": r"annapoorna|annapurna",
}
WIN_LOSS_LABELS = ("win", "loss", "neutral")
MAX_CLASSIFIED = 120
WIN_RE = re.compile(r"better than|best compared|superior to|nicer than|tastier than|prefer(?:red)? (?:this|it) "
                    r"over|instead of|unlike|beats|than (?:the )?(?:usual|other)", re.I)
LOSS_RE = re.compile(r"used to (?:come|go|visit)|now (?:go|going|prefer|we go)|switched to|moved to|"
                     r"better at|better in|\w+ is better|go to .+ instead|not as good as|worse than|"
                     r"inferior to|prefer .+ over", re.I)


def rival_mentions(df: pd.DataFrame, own: str) -> pd.DataFrame:
    """Own-restaurant reviews that name a rival chain (regex name filter)."""
    rows = []
    for rival, pat in RIVAL_PATTERNS.items():
        if re.search(pat, own, re.I):
            continue
        hit = df[(df["restaurant"] == own) & df["review"].str.contains(pat, case=False, regex=True)]
        rows.append(hit.assign(rival=rival))
    return pd.concat(rows) if rows else df.iloc[0:0].assign(rival=pd.Series(dtype=str))


def _rule_label(text: str) -> str:
    if LOSS_RE.search(text) and not re.search(r"better than", text, re.I):
        return "loss"
    return "win" if WIN_RE.search(text) else "neutral"


@st.cache_data(show_spinner="Classifying competitor mentions with Groq…")
def classify_win_loss(pairs: tuple[tuple[str, str, str], ...]) -> tuple[list[str], str]:
    """(own, rival, text) -> win/loss/neutral. Uses Groq LLM when a key is set, else a keyword fallback."""
    try:
        from engine.config import GROQ_API_KEY, GROQ_MODEL
        from engine.replies import _groq
    except ImportError:
        GROQ_API_KEY, GROQ_MODEL, _groq = "", "", None
    if not (GROQ_API_KEY and _groq):
        return [_rule_label(t) for _, _, t in pairs], "Keyword rules (set GROQ_API_KEY for the Groq LLM)"
    labels = []
    for own, rival, text in pairs:
        prompt = (f"A customer review of {own} mentions a competitor, {rival}.\nReview: \"{text[:600]}\"\n"
                  f"Answer with exactly one word. 'win' if the customer says {own} is better than {rival} or "
                  f"chose {own} over it; 'loss' if they say {rival} is better or they left {own} for {rival}; "
                  f"otherwise 'neutral'.")
        out = (_groq(prompt, max_tokens=5, temperature=0) or "").strip().lower()
        labels.append(next((l for l in WIN_LOSS_LABELS if l in out), _rule_label(text)))
    return labels, f"Groq LLM · {GROQ_MODEL} (zero-shot classifier)"


def win_loss_section(df: pd.DataFrame, own: str) -> None:
    """Competitive win/loss: regex name filter -> Groq classify -> monthly net win rate."""
    st.markdown("### Competitive win / loss tracker")
    cols = st.columns(len(WIN_LOSS_STEPS))
    for col, (title, sub) in zip(cols, WIN_LOSS_STEPS):
        col.markdown(f"<div style='border:1px solid #ddd;border-radius:12px;padding:10px;text-align:center;"
                     f"background:#f7f5f0'><b>{title}</b><br><small>{sub}</small></div>", unsafe_allow_html=True)
    men = rival_mentions(df, own)
    if men.empty:
        st.info(f"No {own} reviews name a rival chain in the selected period.")
        return
    men = men.sort_values("review_date", ascending=False).head(MAX_CLASSIFIED)
    labels, engine = classify_win_loss(tuple(zip([own] * len(men), men["rival"], men["review"])))
    men = men.assign(outcome=labels)
    st.caption(f"ML model used: **{engine}**. Net win rate = (wins − losses) / (wins + losses). "
               f"Latest {MAX_CLASSIFIED} rival mentions are classified.")
    wins, losses = (men["outcome"] == "win").sum(), (men["outcome"] == "loss").sum()
    net = (wins - losses) / (wins + losses) * 100 if wins + losses else float("nan")
    kpi_row([("Rival mentions", f"{len(men):,}"), ("Wins", str(wins)), ("Losses", str(losses)),
             ("Net win rate", f"{net:+.0f}%" if pd.notna(net) else "n/a")])
    c1, c2 = st.columns(2)
    by_rival = men.groupby(["rival", "outcome"]).size().reset_index(name="reviews")
    c1.markdown("**Outcome by rival**")
    c1.plotly_chart(px.bar(by_rival, y="rival", x="reviews", color="outcome", orientation="h",
                           color_discrete_map={"win": "#1D9E75", "loss": "#D85A30", "neutral": "#B4B2A9"})
                    .update_layout(margin=dict(t=10, b=10), yaxis_title=None, legend_title=None), width="stretch")
    monthly = men.assign(month=pd.to_datetime(men["review_date"]).dt.to_period("M").dt.to_timestamp())
    monthly = monthly.groupby("month")["outcome"].agg(
        wins=lambda s: (s == "win").sum(), losses=lambda s: (s == "loss").sum()).reset_index()
    monthly = monthly[(monthly["wins"] + monthly["losses"]) > 0]
    monthly["net_win_rate"] = (monthly["wins"] - monthly["losses"]) / (monthly["wins"] + monthly["losses"]) * 100
    c2.markdown("**Net win rate by month**")
    if len(monthly):
        c2.plotly_chart(px.line(monthly, x="month", y="net_win_rate", markers=True)
                        .update_layout(margin=dict(t=10, b=10), yaxis=dict(range=[-100, 100], ticksuffix="%"),
                                       yaxis_title=None, xaxis_title=None), width="stretch")
    else:
        c2.info("Not enough win/loss mentions to chart a monthly trend.")
    g, l = st.columns(2)
    for box, label, colour in ((g, "win", "#E8F5EC"), (l, "loss", "#FBE9EA")):
        ex = men[men["outcome"] == label].head(3)
        box.markdown(f"**Example {label}s**")
        for t, r in zip(ex["review"], ex["rival"]):
            box.markdown(f"<div style='background:{colour};border-radius:8px;padding:8px;margin-bottom:6px'>"
                         f"“{t[:220]}” <small>(vs {r})</small></div>", unsafe_allow_html=True)


DRIVER_STEPS = [("Cluster dummies", "From HDBSCAN"), ("LightGBM", "One per chain"),
                ("SHAP", "Importance per cluster")]
MIN_DRIVER_REVIEWS = 40


@st.cache_data(show_spinner="Training LightGBM + SHAP…")
def driver_importance(clusters: tuple[str, ...], ratings: tuple[float, ...]) -> pd.DataFrame:
    """LightGBM on cluster dummies -> rating; importance = mean |SHAP| per cluster, performance = avg rating."""
    import lightgbm as lgb
    import shap

    d = pd.DataFrame({"cluster": clusters, "rating": ratings})
    keep = d["cluster"].value_counts()
    d = d[d["cluster"].isin(keep[keep >= 5].index)]
    x = pd.get_dummies(d["cluster"]).astype(int)
    model = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=8, min_child_samples=5,
                              verbose=-1, random_state=0).fit(x, d["rating"])
    sv = shap.TreeExplainer(model).shap_values(x)
    imp = pd.Series(abs(sv).mean(axis=0), index=x.columns)
    perf = d.groupby("cluster")["rating"].agg(["mean", "size"])
    return pd.DataFrame({"cluster": imp.index, "importance": imp.values,
                         "performance": perf.loc[imp.index, "mean"].values,
                         "reviews": perf.loc[imp.index, "size"].values})


def driver_matrix_section(df: pd.DataFrame, own: str) -> None:
    """Importance (mean |SHAP|) vs performance (avg rating) quadrant chart for one chain's clusters."""
    st.markdown("### Priority matrix: what to fix first")
    cols = st.columns(len(DRIVER_STEPS))
    for col, (title, sub) in zip(cols, DRIVER_STEPS):
        col.markdown(f"<div style='border:1px solid #ddd;border-radius:12px;padding:10px;text-align:center;"
                     f"background:#f7f5f0'><b>{title}</b><br><small>{sub}</small></div>", unsafe_allow_html=True)
    chains = sorted(df["restaurant"].unique())
    chain = st.selectbox("Chain (one LightGBM model per chain)", chains,
                         index=chains.index(own) if own in chains else 0, key="driver_chain")
    d = df[(df["restaurant"] == chain) & df["rating"].notna()]
    d = d[~d["issue"].astype("string").str.strip().str.lower().isin(NOISE_LABELS)]
    if len(d) < MIN_DRIVER_REVIEWS or d["issue"].nunique() < 3:
        st.info(f"Need at least {MIN_DRIVER_REVIEWS} clustered, rated reviews and 3 clusters for {chain}.")
        return
    try:
        res = driver_importance(tuple(d["issue"].astype(str)), tuple(d["rating"].astype(float)))
    except ImportError:
        st.info("Install 'lightgbm' and 'shap' to show the priority matrix.")
        return
    x_mid, y_mid = res["importance"].median(), d["rating"].mean()
    res["quadrant"] = [("Fix first" if p < y_mid else "Keep strengths") if i >= x_mid
                       else ("Low priority" if p < y_mid else "Maintain")
                       for i, p in zip(res["importance"], res["performance"])]
    fig = px.scatter(res, x="importance", y="performance", text="cluster", color="quadrant", size="reviews",
                     size_max=28, color_discrete_map={"Fix first": "#D85A30", "Keep strengths": "#1D9E75",
                                                      "Maintain": "#3B8BD4", "Low priority": "#B4B2A9"})
    xmax, ymin, ymax = res["importance"].max() * 1.1, res["performance"].min() - 0.2, res["performance"].max() + 0.2
    fig.add_shape(type="rect", x0=x_mid, x1=xmax, y0=ymin, y1=y_mid, fillcolor="#FBE9EA", opacity=0.5, line_width=0,
                  layer="below")
    fig.add_vline(x=x_mid, line_dash="dash", line_color="#999")
    fig.add_hline(y=y_mid, line_dash="dash", line_color="#999")
    for txt, ax, ay in (("Maintain", 0.01, 0.98), ("Keep strengths", 0.99, 0.98),
                        ("Low priority", 0.01, 0.02), ("Fix first", 0.99, 0.02)):
        fig.add_annotation(xref="paper", yref="paper", x=ax, y=ay, text=f"<b>{txt}</b>", showarrow=False,
                           xanchor="left" if ax < 0.5 else "right", yanchor="top" if ay > 0.5 else "bottom")
    fig.update_traces(textposition="top center")
    fig.update_layout(margin=dict(t=10, b=10), height=480, legend_title=None,
                      xaxis=dict(title="Importance (mean |SHAP|) →", range=[0, xmax]),
                      yaxis=dict(title="Performance (avg rating in cluster) ↑", range=[ymin, ymax]))
    st.plotly_chart(fig, width="stretch")
    st.caption("ML model used: **LightGBM regressor** (rating ~ HDBSCAN cluster dummies, one model per chain) "
               "explained with **SHAP TreeExplainer**. Dashed lines = median importance and the chain's average "
               "rating. Bubble size = reviews in the cluster.")
    fix = res[res["quadrant"] == "Fix first"].sort_values("importance", ascending=False)
    if len(fix):
        st.markdown("**Fix first:** " + ", ".join(fix["cluster"]))


def competitor_tab(df: pd.DataFrame, an, issues_ctx: dict) -> None:
    """Head-to-head view of the selected restaurant against the chains chosen in 'Compare with'."""
    own = issues_ctx["restaurant"]
    if df["restaurant"].nunique() < 2:
        st.info("Select at least one competitor in 'Compare with' (sidebar) to unlock the cluster comparison.")
        win_loss_section(df, own)
        driver_matrix_section(df, own)
        return
    rows = [{"restaurant": r, "reviews": len(d), "score": an.reputation_score(d),
             "avg_rating": d["rating"].mean(), "negative_pct": (d["sentiment"] == "negative").mean() * 100}
            for r, d in df.groupby("restaurant")]
    comp = pd.DataFrame(rows).sort_values("score", ascending=False)
    rivals = comp[comp["restaurant"] != own]
    own_row = comp[comp["restaurant"] == own]
    if len(own_row) and len(rivals):
        gap = own_row["score"].iloc[0] - rivals["score"].mean()
        rank = int((comp["score"] > own_row["score"].iloc[0]).sum()) + 1
        kpi_row([(f"{own} reputation score", f"{own_row['score'].iloc[0]:.1f}"),
                 ("Rank among compared chains", f"{rank} of {len(comp)}"),
                 ("Gap vs rival average", f"{gap:+.1f}"),
                 ("Best rival", f"{rivals.iloc[0]['restaurant']} ({rivals.iloc[0]['score']:.1f})")])
    c1, c2 = st.columns(2)
    c1.markdown("**Reputation score by chain**")
    c1.plotly_chart(px.bar(comp.sort_values("score"), y="restaurant", x="score", orientation="h", text="score",
                           range_x=[0, 100]).update_traces(texttemplate="%{text:.1f}")
                    .update_layout(margin=dict(t=10, b=10), yaxis_title=None), width="stretch")
    c2.markdown("**Negative review share by chain**")
    c2.plotly_chart(px.bar(comp.sort_values("negative_pct"), y="restaurant", x="negative_pct", orientation="h",
                           text="negative_pct").update_traces(texttemplate="%{text:.1f}%")
                    .update_layout(margin=dict(t=10, b=10), yaxis_title=None, xaxis_title="% negative"),
                    width="stretch")
    cluster_share_chart(df)
    st.markdown("**Reviews per cluster and chain**")
    clustered = df[~df["issue"].astype("string").str.strip().str.lower().isin(NOISE_LABELS)]
    st.dataframe(pd.crosstab(clustered["issue"], clustered["restaurant"], margins=True, margins_name="Total"),
                 width="stretch")
    win_loss_section(df, own)
    driver_matrix_section(df, own)


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

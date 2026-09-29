"""Analytics: reputation score, KPIs, weekly sentiment, top issues, spikes, priority ranking."""
import numpy as np
import pandas as pd

from .config import HIGH_THRESHOLD, ISSUE_SEVERITY, POSITIVE_ISSUE, SPIKE_MIN_COUNT, SPIKE_RATIO

CLOSED = ("Replied", "Resolved")


def filter_reviews(df, restaurant=None, branches=None, sources=None, start=None, end=None):
    m = pd.Series(True, index=df.index)
    if restaurant:
        m &= df["restaurant"] == restaurant
    if branches:
        m &= df["branch"].isin(branches)
    if sources:
        m &= df["source"].isin(sources)
    if start is not None:
        m &= df["review_date"] >= pd.Timestamp(start)
    if end is not None:
        m &= df["review_date"] <= pd.Timestamp(end)
    return df[m]


def windows(end, days=30):
    """(start, end) of the current window and of the previous window of equal length."""
    end = pd.Timestamp(end).normalize()
    start = end - pd.Timedelta(days=days - 1)
    return (start, end), (start - pd.Timedelta(days=days), start - pd.Timedelta(days=1))


def reputation_score(df) -> float | None:
    """0-100. 60% average rating (1-5 rescaled) + 40% net sentiment (positive share minus negative share)."""
    if df.empty:
        return None
    rating = df["rating"].dropna()
    avg = rating.mean() if len(rating) else 3.0
    pos = (df["sentiment"] == "positive").mean()
    neg = (df["sentiment"] == "negative").mean()
    return round(100 * (0.6 * (avg - 1) / 4 + 0.4 * (1 + pos - neg) / 2), 1)


def pct_change(cur, prev):
    if not prev:
        return None
    return (cur - prev) / prev * 100


def open_escalations(df, as_of):
    esc = df[(df["urgency"] >= HIGH_THRESHOLD) & (~df["status"].isin(CLOSED))]
    overdue = esc[esc["review_date"] < pd.Timestamp(as_of) - pd.Timedelta(days=2)]
    return esc, overdue


def kpis(cur, prev, as_of):
    esc, overdue = open_escalations(cur, as_of)
    s_cur, s_prev = reputation_score(cur), reputation_score(prev)
    n_cur = (cur["sentiment"] == "negative").mean() * 100 if len(cur) else 0
    n_prev = (prev["sentiment"] == "negative").mean() * 100 if len(prev) else None
    return {
        "reviews": len(cur),
        "reviews_change": pct_change(len(cur), len(prev)),
        "score": s_cur,
        "score_delta": None if s_cur is None or s_prev is None else s_cur - s_prev,
        "neg_share": n_cur,
        "neg_delta": None if n_prev is None else n_cur - n_prev,
        "avg_rating": cur["rating"].mean() if len(cur) else None,
        "open_esc": len(esc),
        "overdue": len(overdue),
    }


def weekly_sentiment(df):
    if df.empty:
        return pd.DataFrame(columns=["week", "sentiment", "reviews"])
    w = df.assign(week=df["review_date"].dt.to_period("W-SUN").dt.start_time)
    return w.groupby(["week", "sentiment"]).size().rename("reviews").reset_index()


def monthly_trend(df):
    if df.empty:
        return pd.DataFrame()
    m = df.assign(month=df["review_date"].dt.to_period("M").dt.to_timestamp())
    g = m.groupby("month").agg(reviews=("id", "size"), avg_rating=("rating", "mean"),
                               negative=("sentiment", lambda s: (s == "negative").mean() * 100))
    g["score"] = m.groupby("month").apply(reputation_score, include_groups=False)
    return g.reset_index()


def top_issues(cur, prev, n=None):
    """Complaint issues (negative + neutral reviews) with change vs previous window."""
    def counts(d):
        d = d[(d["sentiment"] != "positive") & (d["issue"] != POSITIVE_ISSUE)]
        return d["issue"].value_counts()
    c, p = counts(cur), counts(prev)
    t = pd.DataFrame({"mentions": c, "previous": p.reindex(c.index).fillna(0).astype(int)})
    t["change_pct"] = np.where(t["previous"] > 0, (t["mentions"] - t["previous"]) / t["previous"].replace(0, 1) * 100, np.nan)
    t = t.sort_values("mentions", ascending=False).reset_index(names="issue")
    return t.head(n) if n else t


def priority_table(cur):
    """Priority = 0.4 x frequency (normalised) + 0.3 x negative share + 0.3 x low rating (5 - avg) / 4."""
    d = cur[cur["issue"] != POSITIVE_ISSUE]
    if d.empty:
        return pd.DataFrame()
    g = d.groupby("issue").agg(frequency=("id", "size"), avg_rating=("rating", "mean"),
                               neg_share=("sentiment", lambda s: (s == "negative").mean()),
                               escalations=("urgency", lambda u: (u >= HIGH_THRESHOLD).sum()))
    g["freq_norm"] = g["frequency"] / g["frequency"].max()
    g["low_rating"] = (5 - g["avg_rating"].fillna(3)) / 4
    g["priority_score"] = (0.4 * g["freq_norm"] + 0.3 * g["neg_share"] + 0.3 * g["low_rating"]) * 100
    g["severity"] = g.index.map(ISSUE_SEVERITY)
    g["priority"] = pd.cut(g["priority_score"], [-1, 45, 65, 101], labels=["Low", "Medium", "High"])
    g = g.sort_values("priority_score", ascending=False).reset_index()
    g.insert(0, "rank", range(1, len(g) + 1))
    return g[["rank", "issue", "frequency", "avg_rating", "neg_share", "escalations", "priority_score", "priority"]]


def spikes(df, as_of):
    """Negative reviews per branch x issue in the last 7 days vs the average of the 4 weeks before."""
    as_of = pd.Timestamp(as_of).normalize()
    neg = df[(df["sentiment"] == "negative") & (df["issue"] != POSITIVE_ISSUE)]
    last = neg[neg["review_date"] > as_of - pd.Timedelta(days=7)]
    base = neg[(neg["review_date"] <= as_of - pd.Timedelta(days=7)) & (neg["review_date"] > as_of - pd.Timedelta(days=35))]
    c = last.groupby(["branch", "issue"]).size().rename("this_week")
    b = (base.groupby(["branch", "issue"]).size() / 4).rename("weekly_avg")
    t = pd.concat([c, b], axis=1).fillna(0)
    t = t[t["this_week"] >= SPIKE_MIN_COUNT]
    t["ratio"] = t["this_week"] / t["weekly_avg"].clip(lower=1)  # baseline of at least 1/week avoids silly ratios
    t = t[t["ratio"] >= SPIKE_RATIO].sort_values("ratio", ascending=False)
    return t.reset_index()


def competitor_scores(df, restaurants, start, end):
    rows = []
    for r in restaurants:
        d = filter_reviews(df, r, start=start, end=end)
        rows.append({"restaurant": r, "score": reputation_score(d), "reviews": len(d),
                     "avg_rating": d["rating"].mean() if len(d) else None})
    return pd.DataFrame(rows).sort_values("score", ascending=False, na_position="last")


def branch_table(cur):
    if cur.empty:
        return pd.DataFrame()
    g = cur.groupby("branch")
    t = pd.DataFrame({
        "reviews": g.size(),
        "avg_rating": g["rating"].mean().round(2),
        "negative_pct": (g["sentiment"].apply(lambda s: (s == "negative").mean()) * 100).round(1),
        "score": g.apply(reputation_score, include_groups=False),
        "top_complaint": g.apply(lambda d: d.loc[d["issue"] != POSITIVE_ISSUE, "issue"].mode().iat[0]
                                 if (d["issue"] != POSITIVE_ISSUE).any() else "-", include_groups=False),
    })
    return t.sort_values("score").reset_index()


def reply_stats(cur):
    """Reply rate / time for negative reviews, and resolved count for escalations."""
    neg = cur[cur["sentiment"] == "negative"]
    replied = neg[neg["replied_at"].notna()]
    hours = (pd.to_datetime(replied["replied_at"]) - replied["review_date"]).dt.total_seconds() / 3600
    esc = cur[cur["urgency"] >= HIGH_THRESHOLD]
    return {
        "reply_rate": len(replied) / len(neg) * 100 if len(neg) else None,
        "avg_reply_hours": hours.median() if len(hours) else None,
        "resolved": int((esc["status"] == "Resolved").sum()),
        "escalations": len(esc),
    }

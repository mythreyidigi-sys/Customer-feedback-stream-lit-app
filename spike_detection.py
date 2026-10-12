"""Component 3 - complaint-spike detection: existing threshold rule vs Z-score.

Unit of analysis: negative, non-positive-issue reviews counted per 7-day bin and per group
(issue, restaurant x issue, or restaurant x branch x issue). Bin 0 is the 7 days ending on `as_of` (date > as_of - 7), bin k is
k weeks earlier - identical to the bins used by `engine.analytics.spikes`, so the two are comparable.

Rules
- THRESHOLD (existing baseline): count >= 3 and count / max(mean of the previous 4 weeks, 1) >= 2.
- Z-SCORE: count >= 3 and z >= 2.0, where z = (count - mean) / sigma over the previous 8 weeks and
  sigma = max(sample std, sqrt(max(mean, 1))). The Poisson-style floor stops tiny, zero-heavy histories
  (std ~ 0) from producing huge z-scores. Needs >= 6 of the 8 history weeks to lie inside the period
  covered by the data; otherwise the group is reported as "pending - insufficient history" (no periods are
  invented, and weeks before the first review date are never zero-filled).
Isolation Forest is deliberately not used: groups have only a handful of weekly observations of one
feature (a count), which is too little for a multivariate outlier model to add anything over a Z-score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POSITIVE_ISSUE = "Overall Positive Experience"
MIN_COUNT = 3
RATIO_THRESHOLD = 2.0
THRESHOLD_BASELINE_WEEKS = 4
Z_THRESHOLD = 2.0
Z_HISTORY_WEEKS = 8
Z_MIN_HISTORY = 6


def complaint_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Negative reviews with a real issue label and a valid review date."""
    out = df[(df["sentiment"] == "negative") & (df["issue"] != POSITIVE_ISSUE) & df["issue"].notna()].copy()
    out["review_date"] = pd.to_datetime(out["review_date"], errors="coerce")
    return out.dropna(subset=["review_date"])


def weekly_counts(df: pd.DataFrame, as_of, level: str = "branch") -> tuple[pd.DataFrame, int]:
    """Counts per group (rows) x week bin (columns 0..n-1, 0 = latest). Also returns how many bins are
    covered by the data (bins older than the first review date are not observable)."""
    as_of = pd.Timestamp(as_of).normalize()
    key = {"issue": ["issue"], "restaurant": ["restaurant", "issue"]}.get(level, ["restaurant", "branch", "issue"])
    all_dates = pd.to_datetime(df["review_date"], errors="coerce").dropna()
    all_dates = all_dates[all_dates <= as_of]
    if all_dates.empty:
        return pd.DataFrame(), 0
    n_bins = int((as_of - all_dates.min().normalize()).days // 7) + 1
    c = complaint_rows(df)
    c = c[c["review_date"] <= as_of]
    c["bin"] = ((as_of - c["review_date"].dt.normalize()).dt.days // 7).astype(int)
    m = c.groupby(key + ["bin"]).size().unstack("bin", fill_value=0)
    return m.reindex(columns=range(n_bins), fill_value=0), n_bins


def evaluate(counts: pd.DataFrame, n_bins: int, week: int = 0) -> pd.DataFrame:
    """Apply both rules to bin `week` (0 = latest) of every group. Returns one row per group with the
    supporting values for each rule."""
    if counts.empty:
        return pd.DataFrame()
    cur = counts[week]
    hist_avail = max(min(n_bins - week - 1, Z_HISTORY_WEEKS), 0)
    base_cols = [week + k for k in range(1, THRESHOLD_BASELINE_WEEKS + 1) if week + k in counts.columns]
    t_base = counts[base_cols].sum(axis=1) / THRESHOLD_BASELINE_WEEKS if base_cols else pd.Series(0.0, index=counts.index)
    # The existing rule divides by 4 even when fewer weeks exist, so mirror that exactly.
    ratio = cur / t_base.clip(lower=1)
    thr_alert = (cur >= MIN_COUNT) & (ratio >= RATIO_THRESHOLD)

    z_cols = [week + k for k in range(1, Z_HISTORY_WEEKS + 1) if week + k in counts.columns]
    hist = counts[z_cols] if z_cols else pd.DataFrame(index=counts.index)
    enough = hist_avail >= Z_MIN_HISTORY
    mean = hist.iloc[:, :hist_avail].mean(axis=1) if hist_avail else pd.Series(np.nan, index=counts.index)
    std = hist.iloc[:, :hist_avail].std(axis=1, ddof=1) if hist_avail > 1 else pd.Series(np.nan, index=counts.index)
    sigma = np.maximum(std.fillna(0), np.sqrt(np.maximum(mean.fillna(0), 1)))
    z = (cur - mean) / sigma
    z_alert = (cur >= MIN_COUNT) & (z >= Z_THRESHOLD) if enough else pd.Series(False, index=counts.index)

    out = counts.index.to_frame(index=False)
    out["count"] = cur.values
    out["threshold_baseline"] = t_base.values
    out["threshold_ratio"] = ratio.values
    out["threshold_alert"] = thr_alert.values
    out["z_baseline_mean"] = mean.values
    out["z_sigma"] = sigma.values
    out["z_score"] = z.values if enough else np.nan
    out["pct_change"] = ((cur - mean) / mean.where(mean > 0)).values * 100 if enough else np.nan
    out["z_alert"] = z_alert.values
    out["z_status"] = "ok" if enough else "pending - insufficient history"
    out["reason"] = [_reason(r) for r in out.itertuples()]
    return out


def _reason(r) -> str:
    bits = []
    if r.threshold_alert:
        bits.append(f"{r.count} reviews is {r.threshold_ratio:.1f}x the 4-week average "
                    f"({r.threshold_baseline:.2f}/wk; rule needs >= {MIN_COUNT} and >= {RATIO_THRESHOLD:g}x)")
    if r.z_alert:
        bits.append(f"Z = {r.z_score:.1f} vs {Z_HISTORY_WEEKS}-week mean {r.z_baseline_mean:.2f} "
                    f"(sigma {r.z_sigma:.2f}; rule needs Z >= {Z_THRESHOLD:g})")
    if not bits and r.z_status != "ok" and r.count >= MIN_COUNT:
        bits.append("Z-score pending - fewer than "
                    f"{Z_MIN_HISTORY} weeks of history")
    return "; ".join(bits)


def current_alerts(df: pd.DataFrame, as_of, level: str = "branch") -> pd.DataFrame:
    """Groups flagged by either rule in the latest week, with an `agreement` column."""
    counts, n_bins = weekly_counts(df, as_of, level)
    res = evaluate(counts, n_bins, 0)
    if res.empty:
        return res
    res = res[res["threshold_alert"] | res["z_alert"]].copy()
    res["agreement"] = np.select([res["threshold_alert"] & res["z_alert"], res["threshold_alert"]],
                                 ["Both rules", "Threshold only"], "Z-score only")
    return res.sort_values(["z_score", "count"], ascending=False, na_position="last").reset_index(drop=True)


def backtest(df: pd.DataFrame, as_of, level: str = "branch") -> pd.DataFrame:
    """Run both rules for every week that has any history; one row per (group, week)."""
    counts, n_bins = weekly_counts(df, as_of, level)
    rows = []
    for w in range(0, max(n_bins - 1, 0)):
        r = evaluate(counts, n_bins, w)
        if r.empty:
            continue
        r["week_end"] = pd.Timestamp(as_of).normalize() - pd.Timedelta(days=7 * w)
        rows.append(r[r["threshold_alert"] | r["z_alert"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def comparison(df: pd.DataFrame, as_of, level: str = "branch") -> dict:
    """How the two rules agree. Agreement is counted only in weeks where the Z-score is evaluable; threshold
    alerts in earlier weeks (Z pending) are reported separately rather than counted as disagreements."""
    counts, n_bins = weekly_counts(df, as_of, level)
    tot = {"both": 0, "threshold_only": 0, "z_only": 0, "weeks_z_evaluable": 0, "weeks_z_pending": 0,
           "threshold_alerts_while_z_pending": 0}
    for w in range(0, max(n_bins - 1, 0)):
        r = evaluate(counts, n_bins, w)
        if r.empty:
            continue
        if (r["z_status"] == "ok").all():
            tot["weeks_z_evaluable"] += 1
            tot["both"] += int((r.threshold_alert & r.z_alert).sum())
            tot["threshold_only"] += int((r.threshold_alert & ~r.z_alert).sum())
            tot["z_only"] += int((~r.threshold_alert & r.z_alert).sum())
        else:
            tot["weeks_z_pending"] += 1
            tot["threshold_alerts_while_z_pending"] += int(r.threshold_alert.sum())
    return {"level": level, "weeks": n_bins, **tot}

LEVELS = ("issue", "restaurant", "branch")


def main(db: str | None = None) -> None:
    """Backtest both rules over the whole dataset and write outputs/spike_detection/*."""
    import json
    import sqlite3
    from pathlib import Path

    here = Path(__file__).resolve().parent
    cands = [Path(db)] if db else [here / "data" / "empathy.db",
                                    here / "scripts_new792026" / "empathy_engine (5)" / "empathy_engine" / "data" / "empathy.db"]
    path = next(p for p in cands if p.is_file() and sqlite3.connect(p).execute("select count(*) from reviews").fetchone()[0])
    df = pd.read_sql("select * from reviews", sqlite3.connect(path))
    as_of = pd.to_datetime(df["review_date"], errors="coerce").max()
    out = here / "outputs" / "spike_detection"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"db": str(path), "as_of": str(as_of.date()), "levels": {}}
    for lv in LEVELS:
        bt = backtest(df, as_of, lv)
        bt.to_csv(out / f"alerts_backtest_{lv}.csv", index=False)
        counts, _ = weekly_counts(df, as_of, lv)
        summary["levels"][lv] = {**comparison(df, as_of, lv), "groups": len(counts),
                                 "max_weekly_count": int(counts.values.max()) if counts.size else 0}
    (out / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
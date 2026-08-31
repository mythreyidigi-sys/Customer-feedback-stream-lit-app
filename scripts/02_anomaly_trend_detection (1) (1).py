"""
02_anomaly_trend_detection.py
--------------------------------
Module 6, Component 2 -- Trend / Spike Detection.

Two-tier design:
  1. Rule-based z-score baseline (kept for explainability in front of a
     viva/evaluator panel -- easy to defend: "a week is flagged if its
     complaint volume is >2 standard deviations above its trailing average").
  2. Isolation Forest cross-check on the same weekly time series, as an
     independent ML validation layer ("rule-based for transparency,
     ML-validated for robustness").

Usage
-----
    python 02_anomaly_trend_detection.py

Swap-in for real data: replace `load_or_generate_reviews()` with your real
classified export (must have review_date, issue_category, chain columns).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import sys
import os

# Add the scripts directory to the path
sys.path.insert(0, os.path.dirname(__file__))

# Import with flexible naming
try:
    from sample_data import generate_reviews_with_spikes
except ImportError:
    from importlib import import_module
    sample_data_module = import_module('sample_data (1)')
    generate_reviews_with_spikes = sample_data_module.generate_reviews_with_spikes

Z_THRESHOLD = 2.0          # weeks with z-score above this are flagged
ROLLING_WINDOW = 6         # trailing weeks used for the rolling mean/std
IFOREST_CONTAMINATION = 0.08  # expected proportion of anomalous weeks


def weekly_issue_counts(df):
    """Pivot classified reviews into a weekly count per issue_category."""
    labeled = df.dropna(subset=["issue_category"]).copy()
    labeled["week"] = labeled["review_date"].dt.to_period("W").apply(lambda p: p.start_time)
    weekly = (
        labeled.groupby(["issue_category", "week"])
        .size()
        .reset_index(name="review_count")
        .sort_values(["issue_category", "week"])
    )
    return weekly


def zscore_flags(weekly, window=ROLLING_WINDOW, threshold=Z_THRESHOLD):
    """Rule-based baseline: flag weeks whose count is `threshold` std devs
    above the trailing rolling mean for that issue category."""
    out = []
    for cat, g in weekly.groupby("issue_category"):
        g = g.sort_values("week").reset_index(drop=True)
        roll_mean = g["review_count"].rolling(window, min_periods=3).mean()
        roll_std = g["review_count"].rolling(window, min_periods=3).std().replace(0, np.nan)
        z = (g["review_count"] - roll_mean) / roll_std
        g["rolling_mean"] = roll_mean
        g["z_score"] = z
        g["rule_flag"] = (z > threshold).fillna(False)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def isolation_forest_flags(weekly, contamination=IFOREST_CONTAMINATION):
    """ML cross-check: fit one Isolation Forest per issue category on
    [review_count, week_over_week_pct_change] and flag the anomalies it
    finds independently of the z-score rule."""
    out = []
    for cat, g in weekly.groupby("issue_category"):
        g = g.sort_values("week").reset_index(drop=True)
        if len(g) < 8:  # not enough history for a meaningful fit
            g["iforest_flag"] = False
            out.append(g)
            continue
        pct_change = g["review_count"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        X = np.column_stack([g["review_count"].values, pct_change.values])
        iforest = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=42
        )
        preds = iforest.fit_predict(X)  # -1 = anomaly, 1 = normal
        g["iforest_flag"] = preds == -1
        out.append(g)
    return pd.concat(out, ignore_index=True)


def detect_spikes(df=None):
    """Main entry point. Returns a weekly-level DataFrame with both the
    rule-based `rule_flag` and the ML `iforest_flag`, plus an `agreement`
    column, and prints an agreement-rate summary."""
    df = df if df is not None else generate_reviews_with_spikes()[0]
    weekly = weekly_issue_counts(df)
    weekly = zscore_flags(weekly)
    iforest_result = isolation_forest_flags(weekly)
    weekly = weekly.merge(
        iforest_result[["issue_category", "week", "iforest_flag"]],
        on=["issue_category", "week"], how="left"
    )
    weekly["both_flag"] = weekly["rule_flag"] & weekly["iforest_flag"]

    n_rule = weekly["rule_flag"].sum()
    n_iforest = weekly["iforest_flag"].sum()
    n_both = weekly["both_flag"].sum()
    agreement_rate = n_both / max(n_rule, 1)
    print(f"Weeks flagged by z-score rule      : {n_rule}")
    print(f"Weeks flagged by Isolation Forest   : {n_iforest}")
    print(f"Weeks flagged by BOTH (high-confidence spike): {n_both}")
    print(f"Rule -> ML agreement rate            : {agreement_rate:.1%}")

    return weekly


if __name__ == "__main__":
    weekly = detect_spikes()

    print("\nTop 10 highest-confidence spikes (flagged by both methods):")
    spikes = weekly[weekly["both_flag"]].sort_values("z_score", ascending=False)
    cols = ["issue_category", "week", "review_count", "rolling_mean", "z_score"]
    print(spikes[cols].head(10).to_string(index=False))

    weekly.to_csv("module6_weekly_spike_flags.csv", index=False)
    print("\nSaved full weekly flag table -> module6_weekly_spike_flags.csv")

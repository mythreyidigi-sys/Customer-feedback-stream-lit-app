"""
03_priority_ranking_xgboost_shap.py
--------------------------------------
Module 6, Component 3 -- Priority-Ranking Regression Model.

Replaces the manually weighted formula (severity x segment x benchmarking)
with an XGBoost regressor whose TARGET is grounded in outcomes: how much did
average rating drop in the 2 weeks following a complaint spike? The model
therefore *learns* which spike characteristics historically mattered, instead
of the weights being hand-set. SHAP is used to explain individual
predictions (i.e., "why is this spike the top priority?") -- the same
XGBoost + SHAP combination already planned for Module 5, so no new tooling.

Pipeline
--------
1. Take the weekly flag table produced by 02_anomaly_trend_detection.py
   (or regenerate it here).
2. Engineer spike-level features (magnitude, volume, category, platform mix,
   rule/ML agreement).
3. Label each spike with the actual rating drop observed in the following
   2 weeks (the "did this matter" ground truth).
4. Train an XGBoost regressor to predict that rating drop from the features
   -> predicted_impact is the new, learned priority score.
5. Explain the ranking with SHAP.

Usage
-----
    python 03_priority_ranking_xgboost_shap.py

Swap-in for real data: once you have >=6 months of real weekly spike history
with matched pre/post rating figures, replace `build_training_table()`'s
call into sample_data with a query against your own review + rating
timeseries -- the feature engineering and model code below don't need to
change.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import sys
import os

# Add the scripts directory to the path
sys.path.insert(0, os.path.dirname(__file__))

# Import with flexible naming
try:
    from sample_data import ISSUE_CATEGORIES, generate_reviews_with_spikes
except ImportError:
    from importlib import import_module
    sample_data_module = import_module('sample_data (1)')
    ISSUE_CATEGORIES = sample_data_module.ISSUE_CATEGORIES
    generate_reviews_with_spikes = sample_data_module.generate_reviews_with_spikes

from importlib import import_module

try:
    _anom = import_module("02_anomaly_trend_detection")
except ImportError:
    _anom = import_module("02_anomaly_trend_detection (1) (1)")


def rating_drop_after(df, issue_category, week, lookahead_weeks=2):
    """Ground-truth outcome: (avg rating in the `lookahead_weeks` after the
    spike week) - (avg rating in the 4 weeks before it), for that issue
    category. Negative = ratings fell after the spike."""
    labeled = df.dropna(subset=["issue_category"])
    sub = labeled[labeled["issue_category"] == issue_category].copy()
    before = sub[(sub["review_date"] >= week - pd.Timedelta(weeks=4)) &
                 (sub["review_date"] < week)]
    after = sub[(sub["review_date"] >= week) &
                (sub["review_date"] < week + pd.Timedelta(weeks=lookahead_weeks))]
    if len(before) < 3 or len(after) < 3:
        return np.nan
    return after["rating"].mean() - before["rating"].mean()


def build_training_table(df=None):
    """Assemble one row per (issue_category, week) with engineered spike
    features + the rating-drop outcome as the regression target."""
    df, _ = generate_reviews_with_spikes() if df is None else (df, None)
    weekly = _anom.detect_spikes(df)

    weekly["outcome_rating_change"] = weekly.apply(
        lambda r: rating_drop_after(df, r["issue_category"], r["week"]), axis=1
    )
    table = weekly.dropna(subset=["outcome_rating_change", "z_score"]).copy()

    # engineered features for the priority model
    table["issue_category_code"] = table["issue_category"].astype("category").cat.codes
    table["volume"] = table["review_count"]
    table["magnitude_z"] = table["z_score"]
    table["pct_above_rolling_mean"] = (
        (table["review_count"] - table["rolling_mean"]) / table["rolling_mean"].replace(0, np.nan)
    ).fillna(0)
    table["rule_and_ml_agree"] = table["both_flag"].astype(int)

    # target: NEGATIVE rating change is bad -> flip sign so higher = higher priority
    table["priority_target"] = -table["outcome_rating_change"]
    return table, df


FEATURES = ["issue_category_code", "volume", "magnitude_z",
            "pct_above_rolling_mean", "rule_and_ml_agree"]


def train_priority_model(table):
    X = table[FEATURES]
    y = table["priority_target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, random_state=42
    )
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"Priority model test MAE (rating-point scale): {mae:.3f}")
    return model, X_train


def explain_and_rank(model, table, X_train, top_n=10):
    X_all = table[FEATURES]
    table = table.copy()
    table["predicted_priority_score"] = model.predict(X_all)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_all)

    # global feature importance (mean |SHAP|)
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES)
    mean_abs_shap = mean_abs_shap.sort_values(ascending=False)
    print("\nGlobal driver importance (mean |SHAP value|):")
    print(mean_abs_shap.to_string())

    # SHAP summary plot -> saved, not shown interactively
    shap.summary_plot(shap_values, X_all, feature_names=FEATURES, show=False)
    plt.tight_layout()
    plt.savefig("module6_shap_summary.png", dpi=150)
    plt.close()
    print("Saved SHAP summary plot -> module6_shap_summary.png")

    ranked = table.sort_values("predicted_priority_score", ascending=False)
    top = ranked.head(top_n)[
        ["issue_category", "week", "review_count", "magnitude_z",
         "predicted_priority_score"]
    ]
    print(f"\nTop {top_n} spikes by LEARNED priority score:")
    print(top.to_string(index=False))

    # per-spike explanation for the #1 priority item, for the business narrative
    top_idx = ranked.index[0]
    row_pos = table.index.get_loc(top_idx)
    contrib = pd.Series(shap_values[row_pos], index=FEATURES).sort_values(key=abs, ascending=False)
    print(f"\nWhy is '{ranked.iloc[0]['issue_category']}' "
          f"(week of {ranked.iloc[0]['week'].date()}) the #1 priority?")
    print(contrib.to_string())

    return ranked


if __name__ == "__main__":
    table, df = build_training_table()
    print(f"Built {len(table)} labeled spike-week training rows across "
          f"{table['issue_category'].nunique()} issue categories.")

    model, X_train = train_priority_model(table)
    ranked = explain_and_rank(model, table, X_train)
    ranked.to_csv("module6_priority_ranked_spikes.csv", index=False)
    print("\nSaved full ranked spike table -> module6_priority_ranked_spikes.csv")

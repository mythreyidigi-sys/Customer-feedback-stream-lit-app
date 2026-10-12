"""Component 1 - XGBoost issue-priority model (training + evaluation + ranking).

Run:  python issue_priority_model.py [--db path\\to\\empathy.db]
Outputs (outputs/xgb_priority/): model.json, preprocessing.joblib, feature_list.json, target_definition.md,
evaluation.csv, priority_ranking.csv, data_audit.csv, panel.csv, run_summary.json

DATA AUDIT FINDING: the review dataset has NO rating column (only text, issue, sentiment, restaurant,
branch, source, review_date). Rating decline therefore cannot be predicted. The target is reframed as a
*sentiment-decline proxy*: the VADER mean compound score of a restaurant x source unit falls from one
calendar month to the next. Review dates for Google are derived from relative dates ("2 months ago"), so only
month-level resolution is used. Results are exploratory (small panel) and are reported as such.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs" / "xgb_priority"
DB_CANDIDATES = [
    HERE / "data" / "empathy.db",
    HERE / "scripts_new792026" / "empathy_engine (5)" / "empathy_engine" / "data" / "empathy.db",
    HERE / "empathy.db",
]
POSITIVE_ISSUE = "Overall Positive Experience"
NOISE_LABELS = {"noise", "unclassified", "uncategorized", "uncategorised", "outlier", "-1", "other", ""}
MIN_REVIEWS = 5          # minimum reviews in a unit-month for its mean sentiment to be trusted
TEST_FROM = "2026-03"    # transitions whose *target month* is >= this form the held-out test period
SHARE_SHIFT = 0.10       # +10 percentage points of one issue category, used for the counterfactual ranking
SEED = 42


def find_db(path: str | None) -> Path:
    if path:
        return Path(path)
    for p in DB_CANDIDATES:
        if p.is_file():
            with sqlite3.connect(p) as con:
                if con.execute("select count(*) from reviews").fetchone()[0] > 0:
                    return p
    raise FileNotFoundError("No populated empathy.db found; pass --db.")


def load_reviews(db: Path) -> pd.DataFrame:
    with sqlite3.connect(db) as con:
        df = pd.read_sql("select * from reviews", con)
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df["month"] = df["review_date"].dt.to_period("M")
    return df


def audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Column-by-column audit of what the dataset can support."""
    wanted = ["text", "rating", "issue", "sentiment", "branch", "restaurant", "review_date", "collected_at"]
    rows = []
    for c in wanted:
        present = c in df.columns
        nn = int(df[c].notna().sum()) if present else 0
        note = {
            "rating": "Empty - no star ratings were captured, so rating decline cannot be modelled.",
            "collected_at": "Not stored; only imported_at (import time) exists, which is not a review attribute.",
            "review_date": "Present; Google dates derive from relative text so month-level only.",
            "issue": "Present; produced by the classifier (issue_source=classifier), not human-coded.",
            "sentiment": "Present; deterministic link to issue labels - not used as a feature (leakage).",
        }.get(c, "")
        if present and c == "branch":
            note = f"{df[c].nunique()} distinct values; some are scraped file names - too noisy to be a unit."
        rows.append({"column": c, "present": present, "non_null": nn, "rows": len(df), "note": note})
    return pd.DataFrame(rows)


def vader_scores(texts: pd.Series) -> np.ndarray:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    return np.array([sia.polarity_scores(str(t))["compound"] for t in texts])


def complaint_issues(df: pd.DataFrame) -> list[str]:
    s = df["issue"].astype("string").str.strip()
    ok = ~s.str.lower().isin(NOISE_LABELS) & (s != POSITIVE_ISSUE) & s.notna()
    return sorted(s[ok].unique())


def build_panel(df: pd.DataFrame, issues: list[str]) -> pd.DataFrame:
    """One row per restaurant x source x month t. Features use only reviews in month t (and earlier via
    the unit's expanding history); the target uses month t+1 only."""
    df = df.dropna(subset=["month"]).copy()
    df["vader"] = vader_scores(df["text"])
    df["is_neg"] = (df["vader"] <= -0.05).astype(float)
    for i in issues:
        df[f"share__{i}"] = (df["issue"] == i).astype(float)
    agg = {"vader": ["mean", "size"], "is_neg": "mean", **{f"share__{i}": "mean" for i in issues}}
    g = df.groupby(["restaurant", "source", "month"]).agg(agg)
    g.columns = ["sent_mean", "n_reviews", "neg_share", *[f"share__{i}" for i in issues]]
    g = g.reset_index().sort_values(["restaurant", "source", "month"])
    g["complaint_share"] = g[[f"share__{i}" for i in issues]].sum(axis=1)

    nxt = g[["restaurant", "source", "month", "sent_mean", "n_reviews"]].copy()
    nxt["month"] = nxt["month"] - 1
    nxt.columns = ["restaurant", "source", "month", "next_sent_mean", "next_n"]
    g = g.merge(nxt, on=["restaurant", "source", "month"], how="left")
    g = g[(g["n_reviews"] >= MIN_REVIEWS) & (g["next_n"] >= MIN_REVIEWS)].copy()
    g["target_decline"] = (g["next_sent_mean"] < g["sent_mean"]).astype(int)
    g["target_month"] = g["month"] + 1
    g["split"] = np.where(g["target_month"].astype(str) >= TEST_FROM, "test", "train")
    return g.reset_index(drop=True)


def feature_columns(issues: list[str], panel: pd.DataFrame) -> tuple[list[str], list[str]]:
    num = ["n_reviews", "sent_mean", "neg_share", "complaint_share", *[f"share__{i}" for i in issues]]
    cat = ["restaurant", "source"]
    return num, cat


def make_preprocessor(num: list[str], cat: list[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


def make_xgb(pos_weight: float):
    from xgboost import XGBClassifier

    return XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=3, reg_lambda=5.0,
                         scale_pos_weight=pos_weight, eval_metric="logloss", random_state=SEED, n_jobs=1)


def metrics(y, p, thr=0.5) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    pred = (p >= thr).astype(int)
    auc = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")
    return {"ROC-AUC": auc, "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0), "F1": f1_score(y, pred, zero_division=0),
            "Accuracy": accuracy_score(y, pred)}


def bootstrap_auc(y, p, n=1000) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(SEED)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) > 1:
            vals.append(roc_auc_score(y[idx], p[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def shifted(X: pd.DataFrame, issue: str, issues: list[str]) -> pd.DataFrame:
    """Add SHARE_SHIFT to one issue share, scaling the other issue shares down proportionally so total
    complaint share stays <= 1 (the remainder is positive/unclassified reviews, also scaled)."""
    X2 = X.copy()
    col = f"share__{issue}"
    others = [f"share__{i}" for i in issues if i != issue]
    X2[col] = (X[col] + SHARE_SHIFT).clip(upper=1.0)
    room = (1 - X2[col]).clip(lower=0)
    cur_others = X[others].sum(axis=1).replace(0, np.nan)
    scale = (room / cur_others).clip(upper=1.0).fillna(1.0)
    for o in others:
        X2[o] = X[o] * scale
    X2["complaint_share"] = X2[[f"share__{i}" for i in issues]].sum(axis=1)
    return X2


def _uplifts(model, pre, X: pd.DataFrame, issues: list[str]) -> np.ndarray:
    base = model.predict_proba(pre.transform(X))[:, 1]
    return np.array([(model.predict_proba(pre.transform(shifted(X, i, issues)))[:, 1] - base).mean()
                     for i in issues])


def rank_issues(model, pre, panel: pd.DataFrame, issues: list[str], num, cat, n_boot: int = 150) -> pd.DataFrame:
    """Priority = mean change in predicted decline probability when one category's share rises by 10pp
    (counterfactual average marginal effect). The 95% interval comes from refitting the preprocessing and
    XGBoost on bootstrap resamples of the panel, so it reflects model uncertainty. The observed decline rate
    in unit-months above vs below the category's median share is a model-free cross-check."""
    feats = num + cat
    X, y = panel[feats], panel["target_decline"].values
    point = _uplifts(model, pre, X, issues)
    rng = np.random.default_rng(SEED)
    boots = []
    while len(boots) < n_boot:
        idx = rng.integers(0, len(panel), len(panel))
        if len(set(y[idx])) < 2:
            continue
        p_b = make_preprocessor(num, cat).fit(X.iloc[idx])
        m_b = make_xgb(float((y[idx] == 0).sum() / max((y[idx] == 1).sum(), 1))).fit(p_b.transform(X.iloc[idx]), y[idx])
        boots.append(_uplifts(m_b, p_b, X, issues))
    boots = np.array(boots)
    rows = []
    for k, i in enumerate(issues):
        sh = panel[f"share__{i}"]
        hi = panel[sh > sh.median()] if sh.nunique() > 1 else panel.iloc[0:0]
        lo = panel.drop(hi.index)
        rows.append({
            "issue": i,
            "avg_risk_uplift_pp": point[k] * 100,
            "ci_low_pp": np.percentile(boots[:, k], 2.5) * 100,
            "ci_high_pp": np.percentile(boots[:, k], 97.5) * 100,
            "observed_decline_rate_high_share": hi["target_decline"].mean() if len(hi) else np.nan,
            "observed_decline_rate_low_share": lo["target_decline"].mean() if len(lo) else np.nan,
            "mean_share": sh.mean(),
            "reviews": int(round((sh * panel["n_reviews"]).sum())),
        })
    r = pd.DataFrame(rows).sort_values("avg_risk_uplift_pp", ascending=False).reset_index(drop=True)
    r["rank"] = np.arange(1, len(r) + 1)
    r["evidence"] = np.select([r["ci_low_pp"] > 0, r["ci_high_pp"] < 0], ["Higher risk", "Lower risk"], "Inconclusive")
    return r[["rank", "issue", "avg_risk_uplift_pp", "ci_low_pp", "ci_high_pp", "evidence",
              "observed_decline_rate_high_share", "observed_decline_rate_low_share", "mean_share", "reviews"]]

TARGET_DOC = """# Component 1 - target definition and feature list

## Data audit (see data_audit.csv)
- No `rating` column exists and `collected_at` is not stored, so **rating decline cannot be predicted**.
- `review_date` exists (Nov 2025 - Apr 2026) but Google dates come from relative text, so only month-level
  resolution is trusted. Per the brief, the task is reframed as a sentiment-decline proxy with an honest
  caveat; it is not a claim about star-rating decline.

## Unit and target
- Unit: restaurant x source x calendar month `t` (min {min_reviews} reviews in both `t` and `t+1`).
- Sentiment score: VADER compound of each review text; unit score = mean over the month.
- **Target `target_decline` = 1 if mean sentiment in month t+1 < mean sentiment in month t**, else 0.

## Features (all computed from month `t` only; nothing from t+1)
{features}

Excluded as leakage: the dataset `sentiment` label (determined by the issue label), `urgency`, `emotion`,
`red_flag`, anything from month t+1.

## Split
Temporal: train = transitions with target month < {test_from}; test = target month >= {test_from}.

## Models
- Baselines: majority class; logistic regression on the same features; naive "current sentiment above its
  training median -> decline" rule (mean reversion).
- XGBoost: shallow trees (depth 2, 150 rounds, strong regularisation) because the panel is small.

## Ranking of issue categories
Average marginal effect: for each complaint category, raise its share by {shift:.0%} points (others scaled
down proportionally) in every unit-month and average the change in the model's predicted decline
probability. A bootstrap 95% CI over units is reported. The observed decline rate for high- vs low-share
unit-months is shown next to it as a model-free check. Categories whose CI includes 0 are flagged weak.
"""


def run(db_path: str | None) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db = find_db(db_path)
    df = load_reviews(db)
    audit = audit_columns(df)
    audit.to_csv(OUT_DIR / "data_audit.csv", index=False)

    issues = complaint_issues(df)
    panel = build_panel(df, issues)
    num, cat = feature_columns(issues, panel)
    feats = num + cat
    train, test = panel[panel["split"] == "train"], panel[panel["split"] == "test"]
    if len(train) < 20 or len(test) < 10 or test["target_decline"].nunique() < 2:
        raise RuntimeError(f"Panel too small for a valid evaluation (train={len(train)}, test={len(test)}).")

    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression

    pre = make_preprocessor(num, cat).fit(train[feats])
    Xtr, Xte = pre.transform(train[feats]), pre.transform(test[feats])
    ytr, yte = train["target_decline"].values, test["target_decline"].values
    pos_w = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    xgb = make_xgb(pos_w).fit(Xtr, ytr)
    logit = LogisticRegression(C=0.5, max_iter=1000).fit(Xtr, ytr)
    dummy = DummyClassifier(strategy="prior").fit(Xtr, ytr)
    med = train["sent_mean"].median()
    p_rule = (test["sent_mean"].values > med).astype(float)

    models = {"Baseline: majority class": np.full(len(yte), float(ytr.mean() >= 0.5)),
              "Baseline: mean-reversion rule": p_rule,
              "Baseline: logistic regression": logit.predict_proba(Xte)[:, 1],
              "XGBoost": xgb.predict_proba(Xte)[:, 1]}
    rows = []
    for name, p in models.items():
        m = metrics(yte, p)
        lo, hi = bootstrap_auc(yte, p) if name != "Baseline: majority class" else (np.nan, np.nan)
        rows.append({"model": name, **m, "AUC_CI_low": lo, "AUC_CI_high": hi,
                     "train_rows": len(train), "test_rows": len(test), "test_positive_rate": yte.mean()})
    ev = pd.DataFrame(rows)
    ev.to_csv(OUT_DIR / "evaluation.csv", index=False)

    # Final model refit on the whole panel for ranking and deployment (evaluation above stays train-only).
    pre_all = make_preprocessor(num, cat).fit(panel[feats])
    yall = panel["target_decline"].values
    final = make_xgb(float((yall == 0).sum() / max((yall == 1).sum(), 1))).fit(pre_all.transform(panel[feats]), yall)
    ranking = rank_issues(final, pre_all, panel, issues, num, cat)
    ranking.to_csv(OUT_DIR / "priority_ranking.csv", index=False)

    final.save_model(str(OUT_DIR / "model.json"))
    joblib.dump(pre_all, OUT_DIR / "preprocessing.joblib")
    (OUT_DIR / "feature_list.json").write_text(json.dumps(
        {"numeric": num, "categorical": cat, "issues": issues, "target": "target_decline"}, indent=2))
    (OUT_DIR / "target_definition.md").write_text(TARGET_DOC.format(
        min_reviews=MIN_REVIEWS, test_from=TEST_FROM, shift=SHARE_SHIFT,
        features="\n".join(f"- `{f}`" for f in feats)), encoding="utf-8")
    panel.assign(month=panel["month"].astype(str), target_month=panel["target_month"].astype(str)).to_csv(
        OUT_DIR / "panel.csv", index=False)
    best_base = ev[ev["model"].str.startswith("Baseline") & (ev["model"] != "Baseline: majority class")]["ROC-AUC"].max()
    xgb_auc = float(ev.loc[ev["model"] == "XGBoost", "ROC-AUC"].iloc[0])
    summary = {"db": str(db), "reviews": len(df), "panel_rows": len(panel), "train_rows": len(train),
               "test_rows": len(test), "issues": issues, "xgb_auc": xgb_auc, "best_baseline_auc": float(best_base),
               "xgb_beats_baselines": bool(xgb_auc > best_base),
               "any_supported_issue": bool((ranking["evidence"] == "Higher risk").any())}
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(audit.to_string(index=False), "\n")
    print(ev.round(3).to_string(index=False), "\n")
    print(ranking.round(3).to_string(index=False))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", help="path to empathy.db (auto-detected if omitted)")
    run(ap.parse_args().db)

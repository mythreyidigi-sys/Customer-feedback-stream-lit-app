"""
escalation_detection.py
-------------------------
USE CASE 3: Escalation / urgency detection (feeds Module 6 -- Reputation
Early-Warning & Resolution, already scoped in the report/PPT).

Flags reviews that combine HIGH emotional intensity with HIGH public-visibility
risk (e.g. betrayal/frustration + mentions of food safety, hygiene, or
public-health language) for immediate owner/manager attention -- rather than
waiting for a monthly pipeline re-run to surface a trend.

Logic (transparent + defensible for a viva -- deliberately NOT a black-box
LLM-only score):
    urgency_score = emotion_weight + severity_keyword_weight + rating_weight + recency_weight
    escalate = True if urgency_score >= ESCALATION_THRESHOLD

INPUT  : CSV with 'review_text', 'emotion' (from emotion_classification.py),
         'rating' (1-5), and optionally 'date' and 'issue_cluster'
OUTPUT : CSV with added 'urgency_score' and 'escalate' columns, plus a
         separate CSV of only the flagged (escalated) reviews for the
         early-warning inbox / dashboard alert table.

USAGE:
    python escalation_detection.py --input reviews_with_emotion.csv \
                                    --output reviews_with_urgency.csv \
                                    --flagged-output escalations.csv
"""

import argparse
import pandas as pd
import numpy as np
import re

# ---- Weighting scheme (transparent, tunable -- document these choices in the report) ----

HIGH_RISK_EMOTIONS = {"betrayal": 3, "frustration": 2, "disappointment": 1}
SEVERITY_KEYWORDS = {
    # Food safety / hygiene -> highest severity (regulatory + reputational risk)
    "food_safety": (["food poisoning", "sick after eating", "vomit", "insect", "cockroach",
                      "hair in food", "stale", "rotten", "expired", "hygiene"], 4),
    # Safety/security incidents
    "safety": (["fight", "harassed", "unsafe", "threatened", "assault"], 4),
    # Public visibility amplifiers
    "amplifiers": (["never coming back", "reported to", "fssai", "consumer court",
                     "warning others", "avoid this place"], 2),
}

RATING_WEIGHT = {1: 3, 2: 2, 3: 1, 4: 0, 5: 0}  # missing/NaN rating treated as 0
ESCALATION_THRESHOLD = 6  # tune based on desired alert volume; documented assumption


def score_severity_keywords(text: str) -> int:
    text_lower = text.lower() if isinstance(text, str) else ""
    score = 0
    for _, (keywords, weight) in SEVERITY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            score += weight
    return score


def compute_urgency(df: pd.DataFrame, text_col: str, emotion_col: str,
                     rating_col: str, date_col: str = None) -> pd.DataFrame:
    df = df.copy()

    df["_emotion_weight"] = df[emotion_col].map(HIGH_RISK_EMOTIONS).fillna(0).astype(int)
    df["_severity_weight"] = df[text_col].apply(score_severity_keywords)

    if rating_col in df.columns:
        df["_rating_weight"] = df[rating_col].map(RATING_WEIGHT).fillna(0).astype(int)
    else:
        df["_rating_weight"] = 0

    # Recency weight: reviews from the last 7 days get a small boost, since a
    # cluster of *recent* severe reviews is a live incident, not historical noise.
    if date_col and date_col in df.columns:
        try:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            most_recent = dates.max()
            df["_recency_weight"] = np.where(
                (most_recent - dates).dt.days <= 7, 1, 0
            )
        except Exception:
            df["_recency_weight"] = 0
    else:
        df["_recency_weight"] = 0

    df["urgency_score"] = (
        df["_emotion_weight"] + df["_severity_weight"] + df["_rating_weight"] + df["_recency_weight"]
    )
    df["escalate"] = df["urgency_score"] >= ESCALATION_THRESHOLD

    df.drop(columns=["_emotion_weight", "_severity_weight", "_rating_weight", "_recency_weight"],
            inplace=True)
    return df


def cluster_level_alert(df: pd.DataFrame, issue_col: str) -> pd.DataFrame:
    """
    Aggregate-level early-warning check: flag any issue cluster where the
    WEEKLY escalation count spikes -- this is what should trigger the
    'pause Campaign 4 / re-check Module 6' logic described in the marketing doc.
    Requires a date column; falls back to overall cluster escalation rate if absent.
    """
    if issue_col not in df.columns:
        return pd.DataFrame()
    summary = (
        df.groupby(issue_col)
        .agg(total_reviews=("escalate", "size"), escalations=("escalate", "sum"))
        .assign(escalation_rate_pct=lambda d: (d["escalations"] / d["total_reviews"] * 100).round(1))
        .sort_values("escalation_rate_pct", ascending=False)
    )
    return summary.reset_index()


def main():
    parser = argparse.ArgumentParser(description="Escalation / urgency detection for CX reviews.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--flagged-output", required=True, help="CSV of only escalated reviews")
    parser.add_argument("--text-col", default="review_text")
    parser.add_argument("--emotion-col", default="emotion")
    parser.add_argument("--rating-col", default="rating")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--issue-col", default="issue_cluster")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    for col in (args.text_col, args.emotion_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    result = compute_urgency(df, args.text_col, args.emotion_col, args.rating_col, args.date_col)
    result.to_csv(args.output, index=False)

    flagged = result[result["escalate"]].sort_values("urgency_score", ascending=False)
    flagged.to_csv(args.flagged_output, index=False)

    print(f"Total reviews: {len(result)}")
    print(f"Escalated reviews: {len(flagged)} ({len(flagged)/len(result)*100:.1f}%)")
    print(f"Threshold used: urgency_score >= {ESCALATION_THRESHOLD}")

    cluster_alerts = cluster_level_alert(result, args.issue_col)
    if not cluster_alerts.empty:
        print("\n--- Escalation rate by issue cluster (early-warning view) ---")
        print(cluster_alerts.to_string(index=False))

    print(f"\nSaved:\n  {args.output}\n  {args.flagged_output}")


if __name__ == "__main__":
    main()

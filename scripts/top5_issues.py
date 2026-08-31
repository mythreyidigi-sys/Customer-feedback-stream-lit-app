import os
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "outputs" / "issues" / "clustered_reviews.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs" / "issues"

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find clustered reviews file: {INPUT_FILE}. "
        "Please run scripts/run_hdbscan.py or scripts/cluster_reviews.py first."
    )

df = pd.read_excel(INPUT_FILE)

cluster_labels = {
    0: "Food Quality & Branch Consistency",
    1: "Service Quality",
    2: "Poor Experience / Hygiene Complaints",
    3: "Ambience & Location",
    4: "Food Quantity & Value for Money",
    5: "Overall Positive Experience",
    6: "Pricing & Staff Performance",
    7: "Brand Reputation & Customer Satisfaction",
    8: "Slow Service & Staff Negligence",
    9: "Food Variety & Fast Service"
}

df["issue"] = df["cluster"].map(cluster_labels)

issue_frequency = (
    df["issue"]
    .value_counts()
    .reset_index()
)

issue_frequency.columns = ["Issue", "Frequency"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
issue_frequency.to_excel(
    OUTPUT_DIR / "issue_frequency.xlsx",
    index=False
)

top5 = issue_frequency.head(5)

top5.to_excel(
    OUTPUT_DIR / "top5_issues.xlsx",
    index=False
)

print("\nTOP 5 ISSUES\n")
print(top5)

print("\nSaved:")
print(OUTPUT_DIR / "issue_frequency.xlsx")
print(OUTPUT_DIR / "top5_issues.xlsx")
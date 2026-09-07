"""
root_cause_emotion_analysis.py
--------------------------------
USE CASE 2: Root-cause x emotion mapping.

Takes the output of emotion_classification.py (which already has an
'issue_cluster' column from your HDBSCAN pipeline, e.g. "Slow Service &
Staff Negligence") and cross-tabs it against the 'emotion' column.

This answers: "Which emotion dominates each business issue?" -- e.g.
frustration clustering around wait times, betrayal clustering around
portion/value complaints, delight clustering around a specific dish.

INPUT  : CSV with 'issue_cluster' and 'emotion' columns
OUTPUT : 
    - A cross-tab CSV (counts + row %) ready to import into Power BI as a matrix visual
    - A heatmap PNG for the report/use-case document
    - Optional: per-chain breakdown if a 'chain' column is present

USAGE:
    python root_cause_emotion_analysis.py --input reviews_with_emotion.csv \
                                           --output-dir ./output \
                                           --issue-col issue_cluster
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def build_crosstab(df: pd.DataFrame, issue_col: str, emotion_col: str = "emotion"):
    counts = pd.crosstab(df[issue_col], df[emotion_col])
    row_pct = counts.div(counts.sum(axis=1), axis=0) * 100
    # Sort issues by total review volume (descending) -- matches Top-5 ranking logic
    order = counts.sum(axis=1).sort_values(ascending=False).index
    return counts.loc[order], row_pct.loc[order]


def plot_heatmap(row_pct: pd.DataFrame, output_path: str):
    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(row_pct))))
    im = ax.imshow(row_pct.values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(row_pct.columns)))
    ax.set_xticklabels(row_pct.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(row_pct.index)))
    ax.set_yticklabels(row_pct.index)

    for i in range(row_pct.shape[0]):
        for j in range(row_pct.shape[1]):
            val = row_pct.values[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                     color="white" if val > row_pct.values.max() * 0.5 else "black", fontsize=8)

    ax.set_title("Emotion Composition per Issue Cluster (row %)")
    fig.colorbar(im, ax=ax, label="% of cluster's reviews")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def dominant_emotion_summary(row_pct: pd.DataFrame) -> pd.DataFrame:
    """For each issue cluster, report the single dominant emotion + its share."""
    dominant = row_pct.idxmax(axis=1)
    share = row_pct.max(axis=1)
    summary = pd.DataFrame({
        "issue_cluster": row_pct.index,
        "dominant_emotion": dominant.values,
        "dominant_emotion_pct": share.round(1).values,
    })
    return summary.sort_values("dominant_emotion_pct", ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Root-cause x emotion cross-tab analysis.")
    parser.add_argument("--input", required=True, help="CSV with issue_cluster + emotion columns")
    parser.add_argument("--output-dir", required=True, help="Directory to write outputs")
    parser.add_argument("--issue-col", default="issue_cluster", help="Column with HDBSCAN issue labels")
    parser.add_argument("--emotion-col", default="emotion", help="Column with emotion labels")
    parser.add_argument("--chain-col", default="chain", help="Optional column for per-chain breakdown")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input)

    for col in (args.issue_col, args.emotion_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    counts, row_pct = build_crosstab(df, args.issue_col, args.emotion_col)

    counts_path = os.path.join(args.output_dir, "issue_emotion_counts.csv")
    pct_path = os.path.join(args.output_dir, "issue_emotion_pct.csv")
    heatmap_path = os.path.join(args.output_dir, "issue_emotion_heatmap.png")
    summary_path = os.path.join(args.output_dir, "dominant_emotion_per_issue.csv")

    counts.to_csv(counts_path)
    row_pct.round(1).to_csv(pct_path)
    plot_heatmap(row_pct, heatmap_path)

    summary = dominant_emotion_summary(row_pct)
    summary.to_csv(summary_path, index=False)

    print("--- Dominant emotion per issue cluster ---")
    print(summary.to_string(index=False))

    # Optional: per-chain cut, useful for restaurant-wise benchmarking in Power BI
    if args.chain_col in df.columns:
        chain_counts = pd.crosstab([df[args.chain_col], df[args.issue_col]], df[args.emotion_col])
        chain_path = os.path.join(args.output_dir, "issue_emotion_by_chain.csv")
        chain_counts.to_csv(chain_path)
        print(f"\nPer-chain breakdown saved: {chain_path}")

    print(f"\nSaved:\n  {counts_path}\n  {pct_path}\n  {heatmap_path}\n  {summary_path}")


if __name__ == "__main__":
    main()

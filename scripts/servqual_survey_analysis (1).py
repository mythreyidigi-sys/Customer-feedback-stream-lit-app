"""
servqual_survey_analysis.py
------------------------------
Section 8.1 -- Mini SERVQUAL survey for triangulation.

Provides:
  1. The standard 5-dimension SERVQUAL question bank (Expectation +
     Perception statements), ready to hand out on paper or load into a form
     tool at the outlet.
  2. score_survey(): turns raw 1-7 Likert responses into per-dimension GAP
     scores (Perception - Expectation; negative = service fell short).
  3. triangulate(): maps each SERVQUAL dimension to the NLP issue
     category/categories it corresponds to, and computes a Spearman rank
     correlation between "how bad the SERVQUAL gap is" and "how frequent the
     matching NLP cluster is" -- this is the actual triangulation evidence
     for the report (Section 8.3).

Usage
-----
    python servqual_survey_analysis.py
This runs on a small synthetic set of 12 respondents so the script is
runnable/testable immediately. Replace `SAMPLE_RESPONSES` with your real
10-15 filled-in survey responses (see servqual_survey_template.xlsx, built
by build_survey_template.py, for the field-collection instrument).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------
# 1. SERVQUAL question bank (5 dimensions x ~4 items each)
# ---------------------------------------------------------------------
SERVQUAL_ITEMS = {
    "Tangibles": [
        "The restaurant's seating and dining area are clean and well-maintained.",
        "Restrooms are clean and well-stocked.",
        "Staff are neatly dressed and presentable.",
        "The restaurant's ambience (lighting, noise level, layout) is pleasant.",
    ],
    "Reliability": [
        "The food quantity/portion matches what was promised on the menu/price.",
        "My order is prepared correctly and consistently, every visit.",
        "The taste and quality of food are consistent across visits.",
        "Billing (in-store or via app) is accurate.",
    ],
    "Responsiveness": [
        "Staff serve food within a reasonable/expected time.",
        "Staff respond quickly when I need something (water, extra items, bill).",
        "Wait time for a table during peak hours is reasonable.",
        "Online/delivery orders arrive within the promised time window.",
    ],
    "Assurance": [
        "Staff are knowledgeable about the menu and can answer my questions.",
        "I feel confident about the hygiene and food-safety standards here.",
        "Staff handle complaints or mistakes competently.",
        "I trust the pricing and billing to be fair and transparent.",
    ],
    "Empathy": [
        "Staff are courteous and polite to me.",
        "Staff pay individual attention to my needs (e.g., dietary requests).",
        "Staff make an effort to make my visit pleasant, not just transactional.",
        "I feel valued as a repeat/regular customer.",
    ],
}

# ---------------------------------------------------------------------
# 2. Mapping: SERVQUAL dimension -> matching NLP issue category/categories
#    (used for triangulation against the cluster frequency ranking)
# ---------------------------------------------------------------------
DIMENSION_TO_ISSUE_CATEGORY = {
    "Tangibles": ["Cleanliness & Restroom Hygiene", "Ambience & Seating"],
    "Reliability": ["Food Quantity & Value for Money", "Billing & Online Ordering Issues"],
    "Responsiveness": ["Slow Service & Staff Negligence", "Food Variety & Fast Service"],
    "Assurance": ["Poor Experience & Food Hygiene Complaints"],
    "Empathy": ["Staff Courtesy & Behaviour", "Service Quality"],
}


def score_survey(responses_df):
    """responses_df: one row per respondent, columns = 'Dimension|item text|E'
    or '...|P' for expectation/perception on a 1-7 Likert scale (see
    generate_sample_responses() for the exact shape expected).

    Returns a DataFrame with one row per dimension: mean_expectation,
    mean_perception, mean_gap (P-E), and n_items.
    """
    records = []
    for dim, items in SERVQUAL_ITEMS.items():
        exp_cols = [f"{dim}|{i}|E" for i in range(len(items))]
        per_cols = [f"{dim}|{i}|P" for i in range(len(items))]
        mean_e = responses_df[exp_cols].values.mean()
        mean_p = responses_df[per_cols].values.mean()
        records.append({
            "dimension": dim,
            "mean_expectation": round(mean_e, 2),
            "mean_perception": round(mean_p, 2),
            "mean_gap": round(mean_p - mean_e, 2),  # negative = shortfall
            "n_items": len(items),
        })
    result = pd.DataFrame(records).sort_values("mean_gap")
    return result


def generate_sample_responses(n_respondents=12, seed=1):
    """Synthetic 1-7 Likert responses for a quick end-to-end test run.
    Replace with real data loaded from the filled survey template."""
    rng = np.random.default_rng(seed)
    # bias some dimensions to have a bigger gap, to make the demo meaningful
    dim_bias = {"Tangibles": -0.3, "Reliability": -1.4, "Responsiveness": -1.1,
                "Assurance": -0.5, "Empathy": -0.6}
    rows = []
    for _ in range(n_respondents):
        row = {}
        for dim, items in SERVQUAL_ITEMS.items():
            for i in range(len(items)):
                exp = int(np.clip(rng.normal(6.0, 0.6), 1, 7))
                per = int(np.clip(rng.normal(6.0 + dim_bias[dim], 0.8), 1, 7))
                row[f"{dim}|{i}|E"] = exp
                row[f"{dim}|{i}|P"] = per
        rows.append(row)
    return pd.DataFrame(rows)


def triangulate(servqual_scores, issue_frequency):
    """Compare SERVQUAL gap severity against NLP cluster frequency.

    servqual_scores  : output of score_survey()
    issue_frequency   : dict {issue_category: review_count} from the top-issue
                        ranking table (Section 6.6 / Appendix)

    Returns a merged DataFrame + the Spearman rank correlation between
    "how negative the SERVQUAL gap is" and "how frequent the matching NLP
    cluster is" (positive correlation with -gap = the two methods agree on
    what's worst).
    """
    rows = []
    for _, r in servqual_scores.iterrows():
        cats = DIMENSION_TO_ISSUE_CATEGORY.get(r["dimension"], [])
        matched_freq = sum(issue_frequency.get(c, 0) for c in cats)
        rows.append({
            "dimension": r["dimension"],
            "mean_gap": r["mean_gap"],
            "matched_issue_categories": ", ".join(cats),
            "matched_nlp_frequency": matched_freq,
        })
    merged = pd.DataFrame(rows).sort_values("mean_gap")

    # rank correlation: more negative gap should correspond to higher NLP frequency
    rho, pval = spearmanr(-merged["mean_gap"], merged["matched_nlp_frequency"])
    print(f"\nSpearman rank correlation (SERVQUAL gap severity vs. NLP cluster "
          f"frequency): rho = {rho:.2f}, p = {pval:.3f}")
    if rho > 0.5:
        print("-> Directionally consistent: dimensions with the worst customer-"
              "perceived gaps also correspond to the most frequent NLP-mined "
              "complaint clusters. This corroborates the clustering findings.")
    else:
        print("-> Limited agreement at this sample size (expected with n=10-15 "
              "respondents) -- report as directional/qualitative triangulation, "
              "not a statistically powered test.")
    return merged, rho


# NLP cluster frequencies from the interim top-issue ranking (Appendix / Section 9)
# -- replace with the final 10-category counts once the full dataset is refreshed.
INTERIM_ISSUE_FREQUENCY = {
    "Food Quantity & Value for Money": 851,
    "Food Variety & Fast Service": 654,
    "Service Quality": 553,
    "Slow Service & Staff Negligence": 525,
    "Poor Experience & Food Hygiene Complaints": 160,
    "Cleanliness & Restroom Hygiene": 140,
    "Ambience & Seating": 130,
    "Billing & Online Ordering Issues": 110,
    "Staff Courtesy & Behaviour": 100,
    "Parking & Accessibility": 90,
}


if __name__ == "__main__":
    responses = generate_sample_responses()
    scores = score_survey(responses)
    print("SERVQUAL dimension scores (from synthetic sample respondents):")
    print(scores.to_string(index=False))

    merged, rho = triangulate(scores, INTERIM_ISSUE_FREQUENCY)
    print("\nTriangulation table:")
    print(merged.to_string(index=False))

    scores.to_csv("servqual_dimension_scores.csv", index=False)
    merged.to_csv("servqual_nlp_triangulation.csv", index=False)
    print("\nSaved -> servqual_dimension_scores.csv, servqual_nlp_triangulation.csv")

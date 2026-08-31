# Module 6 ML Components + SERVQUAL Triangulation — Code Bundle

Implements the three ML components discussed for Module 6 (Reputation
Early-Warning & Resolution), plus the mini-SERVQUAL triangulation described
in Section 8.1 of the report.

## Setup

```bash
pip install -r requirements.txt
```

Every script runs immediately out of the box on **synthetic data**
(`sample_data.py`) so you can test the logic before wiring in your real,
cleaned/labeled review export. Each script's docstring explains exactly what
to change to point it at your real data — the short version:

| File | What it needs from your real pipeline |
|---|---|
| `01_issue_classifier.py` | `cleaned_reviews.xlsx` with `review_text`, `issue_category` (from HDBSCAN + Groq labeling) |
| `02_anomaly_trend_detection.py` | Same file + `review_date`, `chain` |
| `03_priority_ranking_xgboost_shap.py` | Same file + historical `rating` over time, so it can learn actual spike -> rating-drop outcomes |
| `servqual_survey_analysis.py` | Filled-in rows from `servqual_survey_template.xlsx` (see below), plus the final top-issue frequency table |

## Files

- **`sample_data.py`** — synthetic dataset generator matching the project's real schema (6,159 reviews, 10 issue categories, 6 chains, 3 platforms). Also injects realistic weekly complaint spikes for testing components 2 and 3.
- **`01_issue_classifier.py`** — Component 1: trains Random Forest / Logistic Regression / Gradient Boosting on the labeled reviews, picks the best by macro-F1, and exposes `classify_new_reviews()` for real-time scoring of incoming reviews (no need to re-run HDBSCAN each time).
- **`02_anomaly_trend_detection.py`** — Component 2: rule-based z-score spike detection (explainable baseline) cross-checked with an Isolation Forest model on the same weekly time series. Saves `module6_weekly_spike_flags.csv`.
- **`03_priority_ranking_xgboost_shap.py`** — Component 3: engineers spike-level features from Component 2's output, trains an XGBoost regressor to predict the rating-drop outcome of each spike, and uses SHAP to explain the ranking. Saves `module6_priority_ranked_spikes.csv` and `module6_shap_summary.png`.
- **`servqual_survey_analysis.py`** — scores 1–7 Likert SERVQUAL responses into per-dimension gap scores and computes a Spearman rank correlation against the NLP cluster frequency ranking (the actual triangulation test for Section 8.3).
- **`build_survey_template.py`** — generates `servqual_survey_template.xlsx`, a printable/fillable field survey instrument (Instructions + Response Form + analysis-ready Raw Responses tabs) for administering the 10–15 respondent survey at 1–2 outlets.

## Suggested run order

```bash
python 01_issue_classifier.py                 # Component 1
python 02_anomaly_trend_detection.py           # Component 2
python 03_priority_ranking_xgboost_shap.py     # Component 3 (reuses Component 2's logic)
python build_survey_template.py                # generates the fillable xlsx
python servqual_survey_analysis.py             # Section 8 triangulation
```

## Notes for the viva

- The z-score rule is retained deliberately as the transparent baseline; the
  Isolation Forest and XGBoost layers are framed as *validation/robustness*
  additions on top of it, not replacements — this is the "rule-based for
  transparency, ML-validated for robustness" pitch.
- All three Module 6 models reuse algorithms already listed in the project's
  technology stack (Random Forest, Logistic Regression, Gradient Boosting,
  XGBoost, SHAP) — no new tooling was introduced.
- Swap `sample_data.py`'s synthetic generator for your real cleaned/labeled
  export as the very first step once you're ready to run this on actual
  project data; no other code in the three Module 6 scripts needs to change,
  since they're all written against the same column schema.

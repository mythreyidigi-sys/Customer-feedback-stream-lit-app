# Restaurant Review Issue Analysis — Streamlit Dashboard

BITS Pilani MBA Dissertation | Student: N.R. Mythreyi | BITS ID: 2024MB22535

A Streamlit dashboard for mining customer experience issues from restaurant
reviews, covering emotion-layer classification, root-cause analysis,
escalation detection, empathetic reply drafting, SERVQUAL survey
triangulation, and reputation management.

## Setup

```bash
pip install -r requirements.txt
streamlit run main.py
```

Every tab degrades gracefully: if a data file it looks for isn't present in
the repo, that tab shows a warning instead of crashing, so the app always
loads even with a partial dataset.

## Full Pipeline (merge_reviews.py through servqual_survey_analysis.py)

![Full pipeline flowchart](pipeline_flow.png)

| # | Script | Produces | Runs |
|---|---|---|---|
| 1 | `merge_reviews.py` | `data/merged_reviews/all_restaurants_reviews.xlsx` — combines raw scraped reviews from all chains/platforms into one file | Offline only |
| 2 | `clean_reviews.py` | `outputs/cleaned_reviews.xlsx` — deduplicates, normalizes restaurant names, strips empty/placeholder reviews | Offline only |
| 3 | `generate_embeddings.py` | `models/embeddings.pkl` — Sentence-Transformer (all-MiniLM-L6-v2) semantic embeddings for every review | Offline only |
| 4 | `Reduced_dimention.py` | `models/embeddings_reduced.pkl` — UMAP dimensionality reduction ahead of clustering | Offline only |
| 5 | `run_hdbscan.py` | `outputs/issues/hdbscan_clustered_reviews.xlsx` — density-based clustering into issue groups (no predefined cluster count) | Offline only |
| 6 | `Cluster_representative_samples.py` | `outputs/issues/cluster_representative_samples.csv` — the reviews closest to each cluster's centroid, for labeling | Offline only |
| 7 | `Autolabelling.py` | `outputs/issues/cluster_topics_labeled.xlsx` — Groq LLM (with keyword-heuristic fallback) turns numeric cluster IDs into business-readable issue labels | Offline only |
| 8 | `01_issue_classifier.py` | `scripts/issue_classifier.joblib` — trains a supervised classifier (Random Forest / Logistic Regression / Gradient Boosting) on the labeled clusters, for real-time scoring of new reviews | Offline only |
| 9 | `apply_classifier.py` | `outputs/reviews_with_issue_classification.xlsx` — applies that trained classifier to the full review set | Offline only |
| 10 | `emotion_classification.py` | `reviews_with_emotion.csv` — tags each review with a dominant emotion (delight, disappointment, frustration, betrayal, relief, nostalgia, neutral) instead of just positive/negative | Offline **or** live in-app |
| 11 | `root_cause_emotion_analysis.py` | `dominant_emotion_per_issue.csv` (+ a heatmap PNG) — cross-tabs emotion against issue cluster | Offline **or** live in-app |
| 12 | `escalation_detection.py` | `escalations.csv` — transparent urgency score (emotion + severity keywords + rating + recency) flagging reviews needing immediate attention | Offline **or** live in-app |
| 13 | `empathetic_reply_generator.py` | `reviews_with_replies.csv` + a digest `.txt` — drafts issue-specific manager replies for human review/approval | Offline **or** live in-app |
| 14 | `servqual_survey_analysis.py` | `servqual_dimension_scores.csv`, `servqual_nlp_triangulation.csv` — scores a 10-15 respondent SERVQUAL field survey and triangulates it against the NLP issue frequencies | Offline **or** live in-app |

**Stages 1-9** need heavier dependencies (Sentence-Transformers, HDBSCAN,
UMAP, Selenium) that don't belong in a cloud dashboard, and are **not part
of this deployed repo** — they run once, locally, and only their *output*
files get committed here. **Stages 10-14** are lightweight enough to also
run live inside the deployed app itself, via the "Emotion & Escalation" and
"SERVQUAL Survey" tabs — `cx_common.py` is a shared helper required by
stages 10-13 (Groq API calls with automatic keyword/template fallback when
no `GROQ_API_KEY` is set); it has no output file of its own.

The Pipeline Status tab in the app shows this exact 14-stage checklist
live, based on which output files actually exist in the repo.

### Optional: Groq API for real LLM output

By default, stages 10, 12-13 run on lightweight keyword/template heuristics
with zero setup. To use real LLM-generated emotion labels and reply drafts
instead, add a secret in Streamlit Cloud (Settings → Secrets):

```toml
GROQ_API_KEY = "your_key_here"
```

To regenerate stages 1-9, run those scripts in order on your own machine
(or Colab/VS Code), then commit only the resulting output files (not the
scripts) to this repo at the paths shown above — the Issue Classification,
Existing Reviews, and Cluster Analysis tabs in `main.py` will pick them up
automatically on the next reboot.

## App Structure (`main.py`)

| Tab | What it shows |
|---|---|
| Issue Classification | Real-time classification of a new review into an issue category (needs `scripts/issue_classifier.joblib`) |
| Existing Reviews | Browse previously classified reviews (needs `outputs/reviews_with_issue_classification.xlsx`) |
| Anomaly Trends | Weekly complaint-volume spikes (needs `module6_weekly_spike_flags.csv`) |
| Priority Ranking | Learned priority score for complaint spikes + SHAP explanation (needs `module6_priority_ranked_spikes.csv`, `module6_shap_summary.png`) |
| Cluster Analysis | HDBSCAN issue clusters + optional 2D UMAP plot (needs `outputs/issues/cluster_topics_labeled.xlsx`) |
| SERVQUAL Survey | Static results from the last offline run, **plus a live recompute** from real filled-in rows in `servqual_survey_template.xlsx` |
| Survey Template | Download the fillable SERVQUAL field-survey workbook |
| Reputation Management | Sentiment velocity monitoring, predictive risk flags, AI response drafting, AI-search visibility, digital footprint audit, bias/transparency auditing — powered by the `reputation_management/` package |
| Emotion & Escalation | Interactive versions of pipeline stages 10-13 above, runnable directly in the app on demand |
| Pipeline Status | Read-only checklist of all 14 pipeline stages and whether their output files are present in the repo |
| About This Project | Renders this README |

## Required Files at Repo Root

**Code:** `main.py`, `cx_common.py`, `emotion_classification.py`,
`root_cause_emotion_analysis.py`, `escalation_detection.py`,
`empathetic_reply_generator.py`, `servqual_survey_analysis.py`,
`reputation_management/` (folder), `requirements.txt`, `README.md`
(this file is mandatory — the About tab errors without it),
`pipeline_flow.png` (the flowchart embedded above).

**Data (optional, each missing file just shows a warning in its tab):**
`outputs/cleaned_reviews.xlsx`, `outputs/reviews_with_issue_classification.xlsx`,
`outputs/issues/cluster_topics_labeled.xlsx`, `module6_weekly_spike_flags.csv`,
`module6_priority_ranked_spikes.csv`, `module6_shap_summary.png`,
`servqual_dimension_scores.csv`, `servqual_nlp_triangulation.csv`,
`servqual_survey_template.xlsx`, `scripts/issue_classifier.joblib`.

## Notes for the Viva

- Emotion classification deliberately uses a small, fixed, business-relevant
  taxonomy (7 emotions) rather than a large open taxonomy, so results stay
  usable in a Power BI slicer and don't fragment cluster sizes.
- Escalation scoring is a transparent, tunable weighted sum
  (`emotion_weight + severity_keyword_weight + rating_weight + recency_weight`)
  rather than a black-box LLM score — defensible and explainable in front of
  an evaluator panel.
- Reply drafts are always a human-review draft, never auto-posted.
- The SERVQUAL survey is a secondary, corroborative triangulation method —
  the primary evidence base remains the full-scale NLP/clustering pipeline.

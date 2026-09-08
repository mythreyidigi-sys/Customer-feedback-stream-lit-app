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

## Current Pipeline (offline scripts, run locally, then commit the output)

| # | Script | Produces |
|---|---|---|
| 1 | `emotion_classification.py` | `reviews_with_emotion.csv` — tags each review with a dominant emotion (delight, disappointment, frustration, betrayal, relief, nostalgia, neutral) instead of just positive/negative |
| 2 | `root_cause_emotion_analysis.py` | `dominant_emotion_per_issue.csv` (+ a heatmap PNG) — cross-tabs emotion against issue cluster |
| 3 | `escalation_detection.py` | `escalations.csv` — transparent urgency score (emotion + severity keywords + rating + recency) flagging reviews needing immediate attention |
| 4 | `empathetic_reply_generator.py` | `reviews_with_replies.csv` + a digest `.txt` — drafts issue-specific manager replies for human review/approval |
| 5 | `servqual_survey_analysis.py` | `servqual_dimension_scores.csv`, `servqual_nlp_triangulation.csv` — scores a 10-15 respondent SERVQUAL field survey and triangulates it against the NLP issue frequencies |

`cx_common.py` is a shared helper used by stages 1-4 (Groq API calls with
automatic keyword/template fallback when no `GROQ_API_KEY` is set) — it has
no output file of its own. The Pipeline Status tab in the app shows this
checklist live, based on which output files actually exist in the repo.

### Optional: Groq API for real LLM output

By default, stages 1, 3-4 run on lightweight keyword/template heuristics
with zero setup. To use real LLM-generated emotion labels and reply drafts
instead, add a secret in Streamlit Cloud (Settings → Secrets):

```toml
GROQ_API_KEY = "your_key_here"
```

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
| Emotion & Escalation | Interactive versions of pipeline stages 1-4 above, runnable directly in the app on demand |
| Pipeline Status | Read-only checklist of which pipeline stages have output files present in the repo |
| About This Project | Renders this README |

## Required Files at Repo Root

**Code:** `main.py`, `cx_common.py`, `emotion_classification.py`,
`root_cause_emotion_analysis.py`, `escalation_detection.py`,
`empathetic_reply_generator.py`, `servqual_survey_analysis.py`,
`reputation_management/` (folder), `requirements.txt`, `README.md`
(this file is mandatory — the About tab errors without it).

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

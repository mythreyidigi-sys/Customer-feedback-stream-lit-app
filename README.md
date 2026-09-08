# Restaurant Review Issue Analysis — Streamlit Dashboard

BITS Pilani MBA Dissertation | Student: N.R. Mythreyi | BITS ID: 2024MB22535

A 9-tab Streamlit dashboard for mining, triaging, and resolving customer
experience issues from restaurant reviews — from intake through to
resolution — with a global sidebar filter (restaurant chain / platform /
review date range) applied across every tab.

## Setup

```bash
pip install -r requirements.txt
streamlit run main.py
```

## App Structure (`main.py`)

| Tab | What it does |
|---|---|
| Customer Review Intake | Submit a new review (restaurant, branch, source, rating, text); classify it instantly, or add it to the working dataset for this session |
| Existing Reviews | Metrics (cleaned reviews / branches / restaurants / sources) + apply the classifier to the full dataset + Reviews-by-Issue and Reviews-by-Restaurant charts |
| SERVQUAL Survey | Loads a real SERVQUAL response file, computes gap scores per dimension, and triangulates against the live issue-category frequencies |
| Emotion Analysis | Classifies every review into a dominant emotion (delight, disappointment, frustration, betrayal, relief, nostalgia, neutral) |
| Root Cause Analysis | Cross-tabs emotion against issue category as a heatmap, plus a dominant-emotion-per-issue table |
| Escalation Alerts | Transparent urgency scoring (emotion + severity keywords + rating + recency); flags reviews needing immediate attention, plus escalation rate by issue |
| Empathetic Reply | Interactive draft-a-reply tool, issue/emotion/restaurant-aware, always a human-review draft |
| Early-Warning Alerts | Week-over-week complaint-volume spike detection per restaurant/branch/issue, with a platform-breakdown drill-down |
| Resolution Workflow | Tracks corrective action (status, assignee, action taken, date, notes) for every flagged issue from Early-Warning Alerts and Escalation Alerts |

Every tab degrades gracefully: if a required file or a previous step's
output isn't available yet, that tab shows a warning or an info message
instead of crashing, so the app always loads.

## Data Flow

1. **Base dataset**: loaded from `outputs/reviews_with_issue_classification.xlsx` if present, else `outputs/cleaned_reviews.xlsx`. Columns are auto-detected and standardized (`review_text`, `restaurant`, `branch`, `platform`, `rating`, `review_date`, `issue_cluster`) regardless of the exact source column names.
2. **Missing rating/date columns**: if your dataset has no rating or review-date column, the app shows a banner and substitutes a neutral rating (3) and synthetic dates spread over the last 8 weeks, so Early-Warning Alerts and Escalation Alerts still have something meaningful to compute against. Replace with real rating/date columns for accurate results.
3. **Classifier**: loaded from `scripts/issue_classifier.joblib` (also checks `scripts_new792026/issue_classifier.joblib` as a fallback path). Applied on demand from the Existing Reviews tab, and automatically to any new review added via Customer Review Intake.
4. **Emotion / Escalation / Reply**: computed live in the app via `emotion_classification.py`, `escalation_detection.py`, `empathetic_reply_generator.py` (shared helper: `cx_common.py`) — no offline pre-computation required, though these also work if you run them offline and wire in their outputs.
5. **SERVQUAL**: loaded from `Restaurant Survey (Responses)SERVQUAL.xlsx` (or `servqual_survey_template.xlsx`) via `servqual_survey_analysis.py`. If your response file's columns don't match the expected `Dimension|item|E/P` format, the tab shows a preview and a note instead of erroring.
6. **Resolution Workflow**: state is kept in `st.session_state` for the current session only. Streamlit Cloud has no built-in database, so this does **not** persist across app reboots — for real persistence, wire the save/load logic to a Google Sheet, Airtable, or small hosted database.

### Optional: Groq API for real LLM output

By default, emotion classification and reply drafting run on lightweight
keyword/template heuristics with zero setup. To use real LLM-generated
output instead, add a secret in Streamlit Cloud (Settings → Secrets):

```toml
GROQ_API_KEY = "your_key_here"
```

## Required Files at Repo Root

**Code:** `main.py`, `cx_common.py`, `emotion_classification.py`,
`root_cause_emotion_analysis.py`, `escalation_detection.py`,
`empathetic_reply_generator.py`, `servqual_survey_analysis.py`,
`requirements.txt`.

**Data (optional, each missing file just shows a warning/info message):**
`outputs/cleaned_reviews.xlsx` or `outputs/reviews_with_issue_classification.xlsx`,
`scripts/issue_classifier.joblib`,
`Restaurant Survey (Responses)SERVQUAL.xlsx` (or `servqual_survey_template.xlsx`).

## Notes for the Viva

- Emotion classification uses a small, fixed, business-relevant taxonomy
  (7 emotions) rather than a large open taxonomy, so results stay usable
  in a Power BI slicer and don't fragment issue cluster sizes.
- Escalation scoring is a transparent, tunable weighted sum
  (`emotion_weight + severity_keyword_weight + rating_weight + recency_weight`)
  rather than a black-box LLM score — defensible and explainable in front
  of an evaluator panel.
- Reply drafts are always a human-review draft, never auto-posted.
- Early-warning spikes and reputation scores are explicitly labeled as
  illustrative when the underlying dataset has no real rating/date columns.
- Resolution Workflow closes the loop: issue detected (Early-Warning /
  Escalation) → corrective action tracked → status updated — directly
  supporting the report's decision-support framing.

# Restaurant Review Issue Analysis — Streamlit Dashboard

BITS Pilani MBA Dissertation | Student: N.R. Mythreyi | BITS ID: 2024MB22535

An 8-tab Streamlit dashboard for mining, triaging, and resolving customer
experience issues from restaurant reviews — from intake through to
resolution — with a global sidebar filter (restaurant chain / platform /
review date range) applied across every tab.

## Setup

```bash
pip install -r requirements.txt
streamlit run main.py
```

## Staff Access

The public app shows only **Customer Review Intake**. The analysis, alert,
reply, and resolution tabs are available only after a staff member selects
**Staff login** and enters the `STAFF_PASSWORD` configured for the app.

For Streamlit Community Cloud, add the password at **App settings -> Secrets**:

```toml
STAFF_PASSWORD = "set-the-password-provided-by-the-app-owner"
```

For local development, place the same value in `.streamlit/secrets.toml`.
That file is ignored by Git and must not be committed.

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
| Resolution Workflow | Tracks corrective action (status, assignee, action taken, date, notes) for every escalated issue |

Every tab degrades gracefully: if a required file or a previous step's
output isn't available yet, that tab shows a warning or an info message
instead of crashing, so the app always loads.

## Data Flow

1. **Base dataset**: loaded from `outputs/reviews_with_issue_classification.xlsx` if present, else `outputs/cleaned_reviews.xlsx`. Columns are auto-detected and standardized (`review_text`, `restaurant`, `branch`, `platform`, `rating`, `review_date`, `issue_cluster`) regardless of the exact source column names.
2. **Missing rating/date columns**: if your dataset has no rating or review-date column, the app shows a banner and substitutes a neutral rating (3) only. Date-based analysis uses only real dates present in the dataset.
3. **Classifier**: loaded from `scripts/issue_classifier.joblib` (also checks `scripts_new792026/issue_classifier.joblib` as a fallback path). Applied on demand from the Existing Reviews tab, and automatically to any new review added via Customer Review Intake.
4. **Emotion / Escalation / Reply**: computed live in the app via `emotion_classification.py`, `escalation_detection.py`, `empathetic_reply_generator.py` (shared helper: `cx_common.py`) — no offline pre-computation required, though these also work if you run them offline and wire in their outputs.
5. **SERVQUAL**: two supported formats, auto-detected:
   - **Field-survey template** (`Dimension|item|E/P` columns, e.g. from `servqual_survey_template.xlsx`) — scored via `servqual_survey_analysis.py`'s `score_survey()`.
   - **Real Google Forms export** (e.g. `Restaurant_Survey__Responses_SERVQUAL.xlsx`) — questions like "Quality of food [EXPECTATION]" / "[ACTUAL]" are auto-matched by keyword to the 5 SERVQUAL dimensions (`GOOGLE_FORMS_QUESTION_TO_DIMENSION` in `main.py`). Dimensions with no matching question in your form (e.g. Assurance, if your form doesn't ask about it) are correctly omitted rather than shown with fabricated data. If you add/reword questions, update that mapping dict to match.
   Either way, the tab triangulates against the live issue-category frequencies from your working dataset.
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
`scripts/issue_classifier.joblib` (also checks `scripts_new792026/issue_classifier.joblib`),
`Restaurant_Survey__Responses_SERVQUAL.xlsx` (a real Google Forms export) or
`servqual_survey_template.xlsx` (the field-survey template format).

## Notes for the Viva

- Emotion classification uses a small, fixed, business-relevant taxonomy
  (7 emotions) rather than a large open taxonomy, so results stay usable
  in a Power BI slicer and don't fragment issue cluster sizes.
- Escalation scoring is a transparent, tunable weighted sum
  (`emotion_weight + severity_keyword_weight + rating_weight + recency_weight`)
  rather than a black-box LLM score — defensible and explainable in front
  of an evaluator panel.
- Reply drafts are always a human-review draft, never auto-posted.
- Reputation scores are explicitly labeled as illustrative when the
  underlying dataset has no real rating/date columns.
- Resolution Workflow closes the loop: issue detected (Escalation) →
  corrective action tracked → status updated — directly
  supporting the report's decision-support framing.

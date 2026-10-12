# Component 1 - target definition and feature list

## Data audit (see data_audit.csv)
- No `rating` column exists and `collected_at` is not stored, so **rating decline cannot be predicted**.
- `review_date` exists (Nov 2025 - Apr 2026) but Google dates come from relative text, so only month-level
  resolution is trusted. Per the brief, the task is reframed as a sentiment-decline proxy with an honest
  caveat; it is not a claim about star-rating decline.

## Unit and target
- Unit: restaurant x source x calendar month `t` (min 5 reviews in both `t` and `t+1`).
- Sentiment score: VADER compound of each review text; unit score = mean over the month.
- **Target `target_decline` = 1 if mean sentiment in month t+1 < mean sentiment in month t**, else 0.

## Features (all computed from month `t` only; nothing from t+1)
- `n_reviews`
- `sent_mean`
- `neg_share`
- `complaint_share`
- `share__Ambience & Location`
- `share__Brand Reputation & Customer Satisfaction`
- `share__Food Quality & Branch Consistency`
- `share__Food Quantity & Value for Money`
- `share__Food Variety & Fast Service`
- `share__Poor Experience / Hygiene Complaints`
- `share__Pricing & Staff Performance`
- `share__Service Quality`
- `share__Slow Service & Staff Negligence`
- `restaurant`
- `source`

Excluded as leakage: the dataset `sentiment` label (determined by the issue label), `urgency`, `emotion`,
`red_flag`, anything from month t+1.

## Split
Temporal: train = transitions with target month < 2026-03; test = target month >= 2026-03.

## Models
- Baselines: majority class; logistic regression on the same features; naive "current sentiment above its
  training median -> decline" rule (mean reversion).
- XGBoost: shallow trees (depth 2, 150 rounds, strong regularisation) because the panel is small.

## Ranking of issue categories
Average marginal effect: for each complaint category, raise its share by 10% points (others scaled
down proportionally) in every unit-month and average the change in the model's predicted decline
probability. A bootstrap 95% CI over units is reported. The observed decline rate for high- vs low-share
unit-months is shown next to it as a model-free check. Categories whose CI includes 0 are flagged weak.

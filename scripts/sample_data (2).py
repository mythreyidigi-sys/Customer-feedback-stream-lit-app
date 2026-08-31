"""
sample_data.py
---------------
Generates a synthetic dataset that mimics the schema of the project's real
cleaned_reviews file (6,159 reviews, 10 HDBSCAN + Groq-labeled issue
categories, 6 chains, 3 platforms, 108 branches).

USE FOR DEMO/TESTING ONLY. In every script in this folder, replace the call
to `load_or_generate_reviews()` with a direct
`pd.read_excel("cleaned_reviews.xlsx")` (or .csv) pointing at your real,
labeled dataset -- the column names below are exactly what the rest of the
scripts expect, so as long as your real export uses (or is renamed to) the
same column names, no other code needs to change.

Expected real-data schema
--------------------------
review_id       : str/int, unique id
review_text      : str, cleaned review text
chain            : str, one of the 6 restaurant chains
branch           : str, outlet/branch name
platform         : str, one of {Google, Zomato, TripAdvisor}
review_date      : datetime
rating           : int 1-5
issue_category   : str, one of the 10 HDBSCAN+Groq labels (NaN/"" if the
                   review was classified as noise by HDBSCAN)
"""
import numpy as np
import pandas as pd

CHAINS = ["A2B", "Sangeetha", "Saravana Bhavan", "Sree Annapoorna",
          "Namma Veedu Vasantha Bhavan", "Geetham"]
PLATFORMS = ["Google", "Zomato", "TripAdvisor"]

ISSUE_CATEGORIES = [
    "Food Quantity & Value for Money",
    "Food Variety & Fast Service",
    "Service Quality",
    "Slow Service & Staff Negligence",
    "Poor Experience & Food Hygiene Complaints",
    "Parking & Accessibility",
    "Billing & Online Ordering Issues",
    "Ambience & Seating",
    "Staff Courtesy & Behaviour",
    "Cleanliness & Restroom Hygiene",
]

# Rough relative frequencies (interim ranking carried forward + 5 extra
# categories discovered by the final HDBSCAN run) -- purely illustrative.
ISSUE_WEIGHTS = np.array([19.2, 14.8, 12.5, 11.9, 3.6, 8.0, 7.5, 8.5, 7.0, 7.0])
ISSUE_WEIGHTS = ISSUE_WEIGHTS / ISSUE_WEIGHTS.sum()

# A handful of representative phrases per category, used only to make the
# synthetic review_text field mildly realistic for the TF-IDF fallback path.
PHRASE_BANK = {
    "Food Quantity & Value for Money": ["portion was too small", "not worth the price", "quantity was less for the money"],
    "Food Variety & Fast Service": ["limited menu options", "wanted more variety", "quick service but same old items"],
    "Service Quality": ["staff was not attentive", "order was wrong", "service could be better"],
    "Slow Service & Staff Negligence": ["waited forever for food", "staff ignored us", "very slow service today"],
    "Poor Experience & Food Hygiene Complaints": ["found the place unhygienic", "food tasted stale", "overall poor experience"],
    "Parking & Accessibility": ["no parking available", "difficult to find parking", "entrance not wheelchair friendly"],
    "Billing & Online Ordering Issues": ["billing was incorrect", "online order got delayed", "app charged extra"],
    "Ambience & Seating": ["seating was cramped", "ambience was noisy", "not enough seating during peak hours"],
    "Staff Courtesy & Behaviour": ["staff was rude", "waiter was impolite", "not a friendly staff"],
    "Cleanliness & Restroom Hygiene": ["restrooms were dirty", "tables were not cleaned", "cleanliness needs improvement"],
}


def load_or_generate_reviews(n=6159, seed=42):
    """Return a DataFrame with the schema documented above.

    Swap this call for `pd.read_excel("cleaned_reviews.xlsx")` once you point
    the scripts at the real, HDBSCAN + Groq labeled export.
    """
    rng = np.random.default_rng(seed)
    n_labeled = int(n * 5909 / 6159)  # match the project's classified share

    dates = pd.date_range("2025-08-01", "2026-08-01", periods=n)
    chain = rng.choice(CHAINS, size=n)
    platform = rng.choice(PLATFORMS, size=n)
    branch = [f"{c} - Branch {b}" for c, b in zip(chain, rng.integers(1, 19, size=n))]

    issue = rng.choice(ISSUE_CATEGORIES, size=n, p=ISSUE_WEIGHTS)
    issue = issue.astype(object)
    noise_idx = rng.choice(n, size=n - n_labeled, replace=False)
    issue[noise_idx] = None

    text = []
    rating = []
    for cat in issue:
        if cat is None:
            text.append("okay experience nothing special")
            rating.append(int(rng.integers(3, 5)))
        else:
            phrase = rng.choice(PHRASE_BANK[cat])
            text.append(phrase)
            # negative-issue clusters skew ratings down
            rating.append(int(rng.integers(1, 3)))

    df = pd.DataFrame({
        "review_id": [f"R{100000+i}" for i in range(n)],
        "review_text": text,
        "chain": chain,
        "branch": branch,
        "platform": platform,
        "review_date": dates,
        "rating": rating,
        "issue_category": issue,
    })
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def generate_reviews_with_spikes(n=6159, seed=7, n_spike_events=25):
    """Like load_or_generate_reviews(), but deliberately injects a handful of
    genuine weekly complaint spikes per issue category, each followed by a
    dip in average rating over the following 2 weeks for roughly 60% of the
    injected spikes (the rest are "noise" spikes with no downstream rating
    impact). This gives scripts 02 (anomaly detection) and 03 (priority
    ranking) realistic signal to detect/learn from.

    Returns
    -------
    df          : the review-level DataFrame (same schema as
                  load_or_generate_reviews)
    spike_truth : DataFrame of the ground-truth injected spikes, with a
                  `caused_rating_drop` label -- useful to sanity-check
                  script 02/03 output against a known answer during testing.
    """
    df = load_or_generate_reviews(n=n, seed=seed)
    rng = np.random.default_rng(seed)

    weeks = pd.period_range("2025-08-01", "2026-08-01", freq="W")
    spike_rows = []
    extra_reviews = []

    for i in range(n_spike_events):
        cat = rng.choice(ISSUE_CATEGORIES)
        week = rng.choice(weeks[5:-3])  # leave room for trailing window + 2wk lookahead
        causes_drop = rng.random() < 0.6
        n_extra = int(rng.integers(15, 40))

        week_start = week.start_time
        for _ in range(n_extra):
            day_offset = int(rng.integers(0, 7))
            phrase = rng.choice(PHRASE_BANK[cat])
            rating = int(rng.integers(1, 3)) if causes_drop else int(rng.integers(2, 4))
            extra_reviews.append({
                "review_id": f"SPK{len(extra_reviews)}",
                "review_text": phrase,
                "chain": rng.choice(CHAINS),
                "branch": f"{rng.choice(CHAINS)} - Branch {int(rng.integers(1, 19))}",
                "platform": rng.choice(PLATFORMS),
                "review_date": week_start + pd.Timedelta(days=day_offset),
                "rating": rating,
                "issue_category": cat,
            })

        spike_rows.append({
            "issue_category": cat,
            "week": week_start,
            "extra_reviews": n_extra,
            "caused_rating_drop": causes_drop,
        })

    df = pd.concat([df, pd.DataFrame(extra_reviews)], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    spike_truth = pd.DataFrame(spike_rows)
    return df, spike_truth


if __name__ == "__main__":
    df = load_or_generate_reviews()
    print(df.shape)
    print(df["issue_category"].value_counts(dropna=False))

    df2, spikes = generate_reviews_with_spikes()
    print("\nWith injected spikes:", df2.shape, "| spike events:", len(spikes))
    print(spikes.head())

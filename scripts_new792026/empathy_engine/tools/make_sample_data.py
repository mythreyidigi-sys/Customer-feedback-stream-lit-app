"""Creates realistic SAMPLE files (6 months, 6 chains, 3 sources) in data/raw/ so you can try the app
before loading your real scraped data. Each source uses a different file layout on purpose,
to show the import pipeline coping with messy real-world columns.

    python tools/make_sample_data.py
"""
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

random.seed(7)
ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"
END = date(2026, 9, 26)
START = date(2026, 4, 1)

CHAINS = {  # chain: (branches, quality 0-1, weight)
    "Geetham": (["T. Nagar", "Anna Nagar", "Adyar", "Velachery"], 0.62, 1.0),
    "A2B": (["Adyar", "T. Nagar", "Porur", "Tambaram", "OMR"], 0.55, 1.4),
    "Sangeetha": (["Mylapore", "Anna Nagar", "Nungambakkam", "Velachery"], 0.72, 1.2),
    "Saravana Bhavan": (["T. Nagar", "Vadapalani", "Mylapore", "Egmore"], 0.66, 1.3),
    "Sree Annapoorna": (["Anna Nagar", "Kilpauk", "Guindy"], 0.64, 0.7),
    "Namma Veedu Vasantha Bhavan": (["Ashok Nagar", "Chromepet", "Anna Nagar"], 0.60, 0.8),
}
NAME_VARIANTS = {"A2B": ["A2B", "A2B Adyar Ananda Bhavan", "Adyar Ananda Bhavan"],
                 "Saravana Bhavan": ["Hotel Saravana Bhavan", "Saravana Bhavan"],
                 "Namma Veedu Vasantha Bhavan": ["Namma Veedu Vasantha Bhavan", "NVVB"],
                 "Sangeetha": ["Sangeetha Veg Restaurant", "Sangeetha"],
                 "Geetham": ["Geetham", "Geetham Veg Restaurant"],
                 "Sree Annapoorna": ["Sree Annapoorna", "Sri Annapoorna"]}

COMPLAINTS = {
    "value": ["Portion size was quite small for the price we paid, expected more for a full meal.",
              "Pongal portion was half of what we got last month for the same price. Not worth it.",
              "Overpriced for the quantity. Two small idlis for this price is too much.",
              "Mini tiffin quantity has reduced a lot, not value for money anymore."],
    "slow": ["We waited almost 40 minutes for our order and had to ask twice before anyone attended to us.",
             "Very slow service during lunch, waited 30 minutes for a simple meals plate.",
             "Nobody came to take the order for 15 minutes even though the place was half empty."],
    "service": ["Service felt rushed and disorganized during the weekend rush hour.",
                "Wrong order served and the bill had an extra item we didn't order.",
                "They forgot our coffee and the parcel was missing a vada."],
    "quality": ["The taste was noticeably different from the branch we usually visit. Sambar was bland.",
                "Food was cold and the dosa was oily. Quality has gone down.",
                "Chutney tasted sour, not fresh at all."],
    "staff": ["Prices have gone up but the staff attitude hasn't improved. The cashier was rude.",
              "Manager was arrogant when we complained. Very unprofessional staff behaviour."],
    "ambience": ["Seating area was cramped and parking near the outlet was difficult.",
                 "Too crowded and noisy, the AC was not working inside."],
    "reputation": ["Used to be our favourite chain, but recent visits have made us reconsider.",
                   "Not what it used to be. Disappointed with the overall experience, will not visit again."],
    "hygiene": ["Table wasn't cleaned before we sat down and there were flies near the counter. Used to love this place.",
                "Found a hair in the sambar. Hygiene is very poor, disgusting.",
                "Washroom was dirty and the cutlery was not clean. Unhygienic.",
                "Saw a cockroach near the billing counter. Worst experience!!!"],
    "variety": ["Limited menu options in the evening compared to what's shown online.",
                "Half the items on the menu were not available after 8 pm."],
}
POSITIVE = ["Consistently good food and friendly staff, one of our regular spots.",
            "Excellent ghee roast and filter coffee. Quick service even on a Sunday.",
            "Tasty meals, clean place and polite staff. Must visit.",
            "Loved the mini tiffin and the sweets. Great value.",
            "Nice ambience for family dinners, food was delicious.",
            "Good food, reasonable price. Service was quick.",
            "Amazing rava dosa, best in the area."]
NEUTRAL = ["Food was okay, nothing special. Service was average.",
           "Decent place for a quick bite, a bit crowded at lunch.",
           "Average taste, prices are slightly high."]
WEIGHTS = {"value": 18, "slow": 15, "service": 13, "quality": 12, "staff": 10, "ambience": 10,
           "reputation": 9, "hygiene": 5, "variety": 5}


def review(chain, quality, day, branch):
    r = random.random()
    spike = chain == "Geetham" and branch == "T. Nagar" and day > END - timedelta(days=8)
    if spike and random.random() < 0.55:
        return random.choice(COMPLAINTS["hygiene"]), random.choice([1, 1, 2])
    if r < quality:
        return random.choice(POSITIVE), random.choice([4, 5, 5])
    if r < quality + 0.08:
        return random.choice(NEUTRAL), 3
    key = random.choices(list(WEIGHTS), weights=list(WEIGHTS.values()))[0]
    return random.choice(COMPLAINTS[key]), random.choice([1, 2, 2, 3])


def rows_for(source, per_day):
    out = []
    for chain, (branches, q, w) in CHAINS.items():
        for i in range((END - START).days + 1):
            day = START + timedelta(days=i)
            drift = -0.06 if (chain == "Geetham" and day.month == 9) else 0  # Geetham dips in Sept
            n = sum(random.random() < per_day * w for _ in range(len(branches)))
            for _ in range(n):
                br = random.choice(branches)
                text, rating = review(chain, q + drift, day, br)
                out.append((chain, br, day, rating, text))
    return out


def spike_rows():
    """Extra hygiene complaints at Geetham T. Nagar in the final week, so the spike alert has something to find."""
    return [("Geetham", "T. Nagar", END - timedelta(days=random.randint(0, 6)), random.choice([1, 2]),
             random.choice(COMPLAINTS["hygiene"])) for _ in range(7)]


def main():
    for s in ("google", "zomato", "tripadvisor"):
        (ROOT / s).mkdir(parents=True, exist_ok=True)

    # Google: restaurant + branch in one field; ISO dates in xlsx, relative dates ("3 weeks ago") in csv
    g = rows_for("Google", 0.45) + spike_rows()
    gdf = pd.DataFrame([{"name": f"{random.choice(NAME_VARIANTS[c])} - {b}", "reviewer": f"User{random.randint(1, 9999)}",
                         "rating": r, "date": d, "review_text": t} for c, b, d, r, t in g])
    cut = len(gdf) * 2 // 3
    gdf.iloc[:cut].to_excel(ROOT / "google" / "google_reviews_apr_jul.xlsx", index=False)
    rel = gdf.iloc[cut:].copy()
    rel["date"] = [f"{(END - d).days // 7} weeks ago" if (END - d).days >= 14 else f"{(END - d).days} days ago"
                   for d in rel["date"]]
    rel.to_csv(ROOT / "google" / "google_reviews_recent.csv", index=False)

    # Zomato: "Rated 4.0", "14 May 2026", separate locality column
    z = rows_for("Zomato", 0.14)
    pd.DataFrame([{"Restaurant": random.choice(NAME_VARIANTS[c]), "Locality": f"{b}, Chennai", "Rated": f"Rated {r}.0",
                   "Review": t, "Posted On": d.strftime("%d %b %Y")} for c, b, d, r, t in z]
                 ).to_csv(ROOT / "zomato" / "zomato_reviews.csv", index=False)

    # TripAdvisor: bubble ratings (40 = 4 stars), full address, "May 14, 2026"
    t = rows_for("TripAdvisor", 0.05)
    pd.DataFrame([{"restaurant_name": random.choice(NAME_VARIANTS[c]), "location": f"No {random.randint(1, 99)}, Main Road, {b}, Chennai 6000{random.randint(10, 99)}",
                   "bubble_rating": r * 10, "review_date": d.strftime("%B %d, %Y"), "review_title": "Visit",
                   "review_body": x} for c, b, d, r, x in t]).to_csv(ROOT / "tripadvisor" / "tripadvisor_reviews.csv", index=False)
    print(f"Sample files written to {ROOT}: Google {len(g)}, Zomato {len(z)}, TripAdvisor {len(t)} reviews "
          f"({START} to {END}).")


if __name__ == "__main__":
    main()

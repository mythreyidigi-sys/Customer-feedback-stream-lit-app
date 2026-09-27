"""Central settings for Empathy Engine. Edit this file to match your data."""
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"            # put files in raw/google, raw/zomato, raw/tripadvisor
REPORT_DIR = DATA_DIR / "reports"     # generated monthly PDFs
DB_PATH = Path(os.getenv("EMPATHY_DB", DATA_DIR / "empathy.db"))

SOURCES = ["Google", "Zomato", "TripAdvisor"]

# Groq (optional). If GROQ_API_KEY is not set, replies and summaries fall back to templates.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# Restaurant name cleaning: any spelling on the left is stored as the name on the right.
# Add aliases you see in your scraped files.
# ---------------------------------------------------------------------------
CHAIN_ALIASES = {
    "a2b": "A2B",
    "adyar ananda bhavan": "A2B",
    "a2b adyar ananda bhavan": "A2B",
    "sangeetha": "Sangeetha",
    "sangeetha veg restaurant": "Sangeetha",
    "hotel saravana bhavan": "Saravana Bhavan",
    "saravana bhavan": "Saravana Bhavan",
    "hsb": "Saravana Bhavan",
    "sree annapoorna": "Sree Annapoorna",
    "sri annapoorna": "Sree Annapoorna",
    "annapoorna": "Sree Annapoorna",
    "namma veedu vasantha bhavan": "Namma Veedu Vasantha Bhavan",
    "nvvb": "Namma Veedu Vasantha Bhavan",
    "vasantha bhavan": "Namma Veedu Vasantha Bhavan",
    "geetham": "Geetham",
    "geetham veg restaurant": "Geetham",
}

# ---------------------------------------------------------------------------
# Issue categories = the 10 categories from your HDBSCAN + Groq labelling (Section 6, Step 2).
# If your file already has an issue/cluster label column, that label is kept as-is.
# Otherwise the classifier assigns one of these using the keywords below.
# ---------------------------------------------------------------------------
ISSUES = {
    "Food Quantity & Value for Money": [
        "portion", "quantity", "small", "less quantity", "overpriced", "expensive", "value for money",
        "not worth", "costly", "price is high", "half", "tiny", "worth the price",
    ],
    "Slow Service & Staff Negligence": [
        "wait", "waited", "waiting", "slow", "late", "delay", "ignored", "no one attended",
        "nobody came", "minutes", "long time", "took forever",
    ],
    "Service Quality": [
        "service", "rushed", "disorganized", "wrong order", "forgot", "mixed up", "billing",
        "bill", "order was wrong", "missing item",
    ],
    "Food Quality & Branch Consistency": [
        "taste", "tasteless", "stale", "cold", "oily", "salty", "bland", "not fresh", "quality",
        "different from", "undercooked", "sour", "spoiled",
    ],
    "Pricing & Staff Performance": [
        "price hike", "prices have gone up", "rude", "arrogant", "attitude", "impolite", "staff behaviour",
        "staff behavior", "manager", "shouted", "unprofessional",
    ],
    "Ambience & Location": [
        "parking", "crowded", "cramped", "seating", "noisy", "ambience", "ambiance", "ac ", "air condition",
        "hot inside", "location", "space",
    ],
    "Brand Reputation & Customer Satisfaction": [
        "used to be", "used to love", "not what it used", "disappointed", "reconsider", "never again",
        "won't come", "will not visit", "lost its", "gone down", "declined",
    ],
    "Poor Experience / Hygiene Complaints": [
        "hygiene", "dirty", "unclean", "not clean", "flies", "fly", "cockroach", "hair", "insect",
        "smell", "washroom", "toilet", "cutlery", "table wasn't cleaned", "food poisoning", "sick",
    ],
    "Food Variety & Fast Service": [
        "variety", "menu", "options", "not available", "limited", "unavailable", "out of stock",
    ],
    "Overall Positive Experience": [
        "excellent", "great", "amazing", "delicious", "tasty", "loved", "friendly", "good food",
        "awesome", "best", "must visit", "fantastic", "wonderful", "nice",
    ],
}
POSITIVE_ISSUE = "Overall Positive Experience"

# How serious each issue is (1-5). Drives urgency and priority.
ISSUE_SEVERITY = {
    "Poor Experience / Hygiene Complaints": 5,
    "Brand Reputation & Customer Satisfaction": 4,
    "Pricing & Staff Performance": 4,
    "Slow Service & Staff Negligence": 3,
    "Food Quality & Branch Consistency": 3,
    "Food Quantity & Value for Money": 3,
    "Service Quality": 3,
    "Ambience & Location": 2,
    "Food Variety & Fast Service": 2,
    "Overall Positive Experience": 1,
}

# Words that force an escalation regardless of rating.
RED_FLAGS = [
    "food poisoning", "vomit", "hospital", "sick after", "cockroach", "insect", "worm", "hair in",
    "lizard", "fungus", "expired", "police", "consumer court", "fssai", "refund",
]

URGENT_THRESHOLD = 8      # urgency >= 8 -> "Urgent"
HIGH_THRESHOLD = 6        # urgency >= 6 -> "High" (goes to Action center)
SPIKE_RATIO = 2.0         # weekly negative count vs trailing 4-week average
SPIKE_MIN_COUNT = 3       # ignore spikes smaller than this

TEAM = ["Unassigned", "Quality team", "Kitchen head", "Branch manager", "ORM team", "Operations head"]

"""Classify review issues with the trained model and a rule-based safety net."""
import re
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from .config import ISSUES, ISSUE_SEVERITY, POSITIVE_ISSUE, RED_FLAGS

ISSUE_MODEL_PATH = Path(__file__).resolve().parents[4] / "scripts_new792026" / "issue_classifier.joblib"
MODEL_CONFIDENCE_THRESHOLD = 0.65
MODEL_LABEL_MAP = {
    "Ambience & Seating": "Ambience & Location",
    "Billing & Online Ordering Issues": "Service Quality",
    "Cleanliness & Restroom Hygiene": "Poor Experience / Hygiene Complaints",
    "Food Quantity & Value for Money": "Food Quantity & Value for Money",
    "Food Variety & Fast Service": "Food Variety & Fast Service",
    "Parking & Accessibility": "Ambience & Location",
    "Poor Experience & Food Hygiene Complaints": "Poor Experience / Hygiene Complaints",
    "Service Quality": "Service Quality",
    "Slow Service & Staff Negligence": "Slow Service & Staff Negligence",
    "Staff Courtesy & Behaviour": "Pricing & Staff Performance",
}

NEG_WORDS = ["bad", "worst", "poor", "pathetic", "horrible", "terrible", "disgusting", "awful", "rude",
             "dirty", "cold", "stale", "slow", "not good", "disappointed", "disappointing", "never again",
             "overpriced", "waste", "tasteless", "cheated", "unhygienic", "not worth", "avoid", "useless"]
POS_WORDS = ["good", "great", "excellent", "amazing", "delicious", "tasty", "loved", "love", "friendly",
             "awesome", "best", "nice", "fantastic", "wonderful", "perfect", "recommend", "fresh", "quick"]

EMOTIONS = {
    "anger": ["worst", "pathetic", "disgusting", "horrible", "cheated", "fraud", "angry", "furious",
              "shameful", "rude", "never again", "!!!", "unacceptable", "ridiculous"],
    "frustration": ["waited", "waiting", "again", "twice", "no one", "nobody", "ignored", "kept asking",
                    "still", "forever", "had to ask", "annoying", "irritating"],
    "disappointment": ["disappointed", "disappointing", "expected", "used to", "not as", "sad", "let down",
                       "not what it used", "declined", "gone down", "reconsider"],
    "delight": ["loved", "amazing", "excellent", "delicious", "fantastic", "wonderful", "awesome",
                "best", "must visit", "superb"],
}


def _rx(words):
    return re.compile(r"(?<![a-z])(" + "|".join(re.escape(w.strip()) for w in words) + r")(?![a-z])")


ISSUE_RX = {k: _rx(v) for k, v in ISSUES.items()}
NEG_RX, POS_RX, RED_RX = _rx(NEG_WORDS), _rx(POS_WORDS), _rx(RED_FLAGS)
EMO_RX = {k: re.compile("|".join(re.escape(w) for w in v)) for k, v in EMOTIONS.items()}


def sentiment_of(text: str, rating) -> str:
    if pd.notna(rating):
        return "positive" if rating >= 4 else "negative" if rating <= 2 else "neutral"
    neg, pos = len(NEG_RX.findall(text)), len(POS_RX.findall(text))
    return "negative" if neg > pos else "positive" if pos > neg else "neutral"


def issue_of(text: str, sentiment: str, red_flag: bool) -> str:
    hits = {k: len(rx.findall(text)) for k, rx in ISSUE_RX.items()}
    neg_hits = {k: v for k, v in hits.items() if k != POSITIVE_ISSUE and v > 0}
    if sentiment == "positive" and not red_flag:
        if len(NEG_RX.findall(text)) <= len(POS_RX.findall(text)) or not neg_hits:
            return POSITIVE_ISSUE
    if red_flag and hits["Poor Experience / Hygiene Complaints"]:
        return "Poor Experience / Hygiene Complaints"
    if neg_hits:
        # most keyword hits wins; ties go to the more severe issue
        return max(neg_hits, key=lambda k: (neg_hits[k], ISSUE_SEVERITY[k]))
    if sentiment == "positive":
        return POSITIVE_ISSUE
    return "Brand Reputation & Customer Satisfaction"  # unhappy but unspecific


def emotion_of(text: str, sentiment: str) -> str:
    if sentiment == "positive":
        return "delight" if EMO_RX["delight"].search(text) else "neutral"
    for emo in ("anger", "frustration", "disappointment"):
        if EMO_RX[emo].search(text):
            return emo
    return "neutral"


def urgency_of(rating, sentiment, issue, emotion, red_flag) -> int:
    base = {1: 6, 2: 5, 3: 3, 4: 1, 5: 1}.get(int(rating) if pd.notna(rating) else 0,
                                          5 if sentiment == "negative" else 2)
    if sentiment != "positive":
        base += ISSUE_SEVERITY.get(issue, 3) - 2
        base += {"anger": 2, "frustration": 1, "disappointment": 1}.get(emotion, 0)
    if red_flag:
        base += 4
    return int(max(1, min(10, base)))


@lru_cache(maxsize=1)
def _load_issue_model(model_path: str, modified_ns: int):
    path = Path(model_path)
    if not path.is_file():
        return None
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not {"model", "vectorizer"} <= bundle.keys():
        raise ValueError(f"Unsupported issue-classifier bundle: {path}")
    model, vectorizer = bundle["model"], bundle["vectorizer"]
    model_classes = set(getattr(model, "classes_", ()))
    if not model_classes or model_classes - MODEL_LABEL_MAP.keys():
        raise ValueError(f"Issue-model labels do not match the configured label map: {path}")
    if vectorizer is None or not hasattr(model, "predict_proba"):
        raise ValueError(f"Issue model must include a vectorizer and predict_proba: {path}")
    return model, vectorizer


def _issue_model():
    modified_ns = ISSUE_MODEL_PATH.stat().st_mtime_ns if ISSUE_MODEL_PATH.is_file() else 0
    return _load_issue_model(str(ISSUE_MODEL_PATH), modified_ns)


def issue_model_status() -> str:
    if _issue_model() is None:
        return f"Trained issue model not found at {ISSUE_MODEL_PATH}; rule-based classification is active."
    return f"Trained issue model loaded from {ISSUE_MODEL_PATH.name}."


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """Classify reviews, using the trained model when confidence is sufficient."""
    df = df.copy()
    if df.empty:
        return df.assign(issue_source=None, issue_confidence=None, sentiment=None, emotion=None,
                         urgency=None, red_flag=None)
    if "issue" not in df.columns:
        df["issue"] = None
    df["issue_source"] = df["issue"].apply(lambda v: "file" if isinstance(v, str) and v.strip() else "classifier")
    out, model_rows = [], []
    for text, rating, issue, src in zip(df["text_clean"], df["rating"], df["issue"], df["issue_source"]):
        red = bool(RED_RX.search(text))
        sent = sentiment_of(text, rating)
        confidence = None
        if src == "file":
            iss, src = issue.strip(), "file"
        elif red:
            iss, src = issue_of(text, sent, red), "rules"
        elif sent == "positive":
            iss, src = POSITIVE_ISSUE, "rules"
        else:
            model_rows.append((len(out), text, sent, rating, red))
            out.append(None)
            continue
        emo = emotion_of(text, sent)
        out.append((iss, src, confidence, sent, emo, urgency_of(rating, sent, iss, emo, red), int(red)))

    if model_rows:
        bundle = _issue_model()
        if bundle is None:
            probabilities = None
        else:
            model, vectorizer = bundle
            probabilities = model.predict_proba(vectorizer.transform([row[1] for row in model_rows]))
        for i, (position, text, sent, rating, red) in enumerate(model_rows):
            if probabilities is None:
                iss, src, confidence = issue_of(text, sent, red), "rules", None
            else:
                best = int(probabilities[i].argmax())
                confidence = float(probabilities[i][best])
                model_issue = MODEL_LABEL_MAP[str(model.classes_[best])]
                if confidence >= MODEL_CONFIDENCE_THRESHOLD:
                    iss, src = model_issue, "ml_model"
                else:
                    iss, src = issue_of(text, sent, red), "rules_low_confidence"
            emo = emotion_of(text, sent)
            out[position] = (iss, src, confidence, sent, emo, urgency_of(rating, sent, iss, emo, red), int(red))
    df[["issue", "issue_source", "issue_confidence", "sentiment", "emotion", "urgency", "red_flag"]] = (
        pd.DataFrame(out, index=df.index)
    )
    return df

"""Classifier: issue category, sentiment, emotion and urgency (1-10) for every review.

Two routes to the issue label:
  1. Your file already has an issue / cluster label (from your HDBSCAN + Groq pipeline) -> kept as-is.
  2. Otherwise a transparent keyword classifier assigns one of the 10 categories in config.ISSUES.
     (tools/discover_issues.py can re-label everything with HDBSCAN later.)
"""
import re

import pandas as pd

from .config import ISSUES, ISSUE_SEVERITY, POSITIVE_ISSUE, RED_FLAGS

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


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """Expects columns text_clean, rating and optional issue (pre-labelled). Adds the classifier columns."""
    df = df.copy()
    if df.empty:
        return df.assign(issue_source=None, sentiment=None, emotion=None, urgency=None, red_flag=None)
    if "issue" not in df.columns:
        df["issue"] = None
    df["issue_source"] = df["issue"].apply(lambda v: "file" if isinstance(v, str) and v.strip() else "classifier")
    out = []
    for text, rating, issue, src in zip(df["text_clean"], df["rating"], df["issue"], df["issue_source"]):
        red = bool(RED_RX.search(text))
        sent = sentiment_of(text, rating)
        iss = issue.strip() if src == "file" else issue_of(text, sent, red)
        emo = emotion_of(text, sent)
        out.append((iss, sent, emo, urgency_of(rating, sent, iss, emo, red), int(red)))
    df[["issue", "sentiment", "emotion", "urgency", "red_flag"]] = pd.DataFrame(out, index=df.index)
    return df

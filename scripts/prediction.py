"""
prediction.py
=============
1. RiskPredictor        -- flags likely reputation risks *before* they go viral,
                            based on early-warning patterns (clusters of similar
                            complaints + accelerating sentiment velocity).
2. CompetitiveBenchmark -- tracks your sentiment vs. competitors on the same
                            topics, not just in isolation.

Clustering uses simple keyword/topic grouping here (swap in your existing
HDBSCAN + Sentence-Transformers pipeline from the CX project -- the interface
is identical: text in, cluster_label + issue_frequency out).
"""

from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from .monitoring import Mention


# A minimal topic lexicon standing in for your HDBSCAN cluster labels.
# In production: call your existing embeddings+HDBSCAN pipeline and pass the
# resulting cluster label into RiskPredictor.score() as `topic`.
_TOPIC_KEYWORDS = {
    "food_quality": ["stale", "cold food", "tasteless", "quality"],
    "wait_time": ["wait", "slow", "delay", "late", "queue"],
    "staff_behavior": ["rude", "staff", "unprofessional", "negligent"],
    "hygiene": ["dirty", "unhygienic", "hygiene", "insect", "smell"],
    "billing": ["overcharge", "bill", "refund", "price"],
}


@dataclass
class RiskFlag:
    topic: str
    severity: str          # "watch" | "elevated" | "critical"
    risk_score: float      # 0..1
    supporting_mentions: int
    rationale: str


class RiskPredictor:
    """
    Combines: (a) cluster size/frequency of a complaint topic,
              (b) how negative it skews,
              (c) how fast it's accelerating (sentiment velocity),
    into a single early-warning risk score, well before volume is large
    enough to "go viral."
    """

    def __init__(self, watch_threshold: float = 0.35,
                 elevated_threshold: float = 0.6,
                 critical_threshold: float = 0.8):
        self.thresholds = {
            "watch": watch_threshold,
            "elevated": elevated_threshold,
            "critical": critical_threshold,
        }

    @staticmethod
    def classify_topic(text: str) -> str:
        lowered = text.lower()
        for topic, keywords in _TOPIC_KEYWORDS.items():
            if any(k in lowered for k in keywords):
                return topic
        return "other"

    def score(self, mentions: List[Mention], velocity_by_bucket: Optional[List[dict]] = None) -> List[RiskFlag]:
        by_topic: Dict[str, List[Mention]] = defaultdict(list)
        for m in mentions:
            by_topic[self.classify_topic(m.text)].append(m)

        # Global average velocity acts as an accelerant multiplier for every topic
        avg_velocity = 0.0
        if velocity_by_bucket:
            recent = velocity_by_bucket[-3:]
            avg_velocity = sum(b["sentiment_velocity"] for b in recent) / max(len(recent), 1)

        flags = []
        total_mentions = max(len(mentions), 1)
        for topic, group in by_topic.items():
            if topic == "other":
                continue
            volume_share = len(group) / total_mentions
            negative_share = sum(1 for m in group if m.nlp and m.nlp.polarity_label == "negative") / len(group)
            velocity_boost = max(avg_velocity, 0) * 2  # only accelerating negativity raises risk

            risk_score = min(1.0, 0.5 * volume_share * 3 + 0.4 * negative_share + velocity_boost)
            severity = self._severity_for(risk_score)
            if severity is None:
                continue

            flags.append(RiskFlag(
                topic=topic,
                severity=severity,
                risk_score=round(risk_score, 2),
                supporting_mentions=len(group),
                rationale=(
                    f"{len(group)} mentions ({volume_share:.0%} of volume), "
                    f"{negative_share:.0%} negative, "
                    f"sentiment velocity {avg_velocity:+.2f} over recent buckets"
                ),
            ))
        return sorted(flags, key=lambda f: f.risk_score, reverse=True)

    def _severity_for(self, score: float) -> Optional[str]:
        if score >= self.thresholds["critical"]:
            return "critical"
        if score >= self.thresholds["elevated"]:
            return "elevated"
        if score >= self.thresholds["watch"]:
            return "watch"
        return None


class CompetitiveBenchmark:
    """
    Compares your brand's sentiment on shared topics against named competitors.
    Feed it Mention lists per brand (each brand's own monitoring stream).
    """

    def __init__(self):
        self._brand_mentions: Dict[str, List[Mention]] = {}

    def add_brand_mentions(self, brand: str, mentions: List[Mention]) -> None:
        self._brand_mentions.setdefault(brand, []).extend(mentions)

    def compare(self, topics: Optional[List[str]] = None) -> List[dict]:
        topics = topics or list(_TOPIC_KEYWORDS.keys())
        rows = []
        for brand, mentions in self._brand_mentions.items():
            by_topic = defaultdict(list)
            for m in mentions:
                by_topic[RiskPredictor.classify_topic(m.text)].append(m)
            for topic in topics:
                group = by_topic.get(topic, [])
                if not group:
                    continue
                avg_polarity = sum(m.nlp.polarity_score for m in group if m.nlp) / len(group)
                rows.append({
                    "brand": brand,
                    "topic": topic,
                    "mention_count": len(group),
                    "avg_polarity": round(avg_polarity, 3),
                })
        return sorted(rows, key=lambda r: (r["topic"], -r["mention_count"]))

    def rank_on_topic(self, topic: str) -> List[dict]:
        return [r for r in self.compare([topic]) if r["topic"] == topic]

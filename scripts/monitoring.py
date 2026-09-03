"""
monitoring.py
=============
Real-time cross-platform mention tracking with "Sentiment Velocity":
how FAST negative sentiment is accelerating, not just its current level.

Plug in real collectors under `SourceClient` (Twitter/X API, Google Reviews
scraper, NewsAPI, Reddit/forum API, etc.) -- everything downstream
(scoring, velocity, anomaly alerts) works on the normalized `Mention` object.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import mean, pstdev
from typing import Callable, Dict, List, Optional

from .nlp_engine import NLPEngine, NLPResult


@dataclass
class Mention:
    id: str
    source: str                 # "google_reviews" | "twitter" | "news" | "forum" | ...
    text: str
    author: Optional[str]
    timestamp: datetime
    url: Optional[str] = None
    nlp: Optional[NLPResult] = None  # filled in by MentionMonitor.ingest()


class SourceClient:
    """
    Stub for a real external data source.
    Replace `fetch_new_mentions` with an actual API/scraper call.
    Must return a list of dicts: {id, source, text, author, timestamp, url}
    """
    def __init__(self, source_name: str, fetch_fn: Optional[Callable[[], List[dict]]] = None):
        self.source_name = source_name
        self._fetch_fn = fetch_fn or (lambda: [])

    def fetch_new_mentions(self) -> List[dict]:
        return self._fetch_fn()


class MentionMonitor:
    """
    Ingests mentions from any number of sources, scores them with NLPEngine,
    and computes Sentiment Velocity + anomaly alerts over a rolling window.
    """

    def __init__(self, nlp_engine: Optional[NLPEngine] = None,
                 velocity_window_hours: int = 24,
                 anomaly_z_threshold: float = 2.0):
        self.nlp = nlp_engine or NLPEngine()
        self.sources: Dict[str, SourceClient] = {}
        self.mentions: List[Mention] = []
        self.velocity_window = timedelta(hours=velocity_window_hours)
        self.anomaly_z_threshold = anomaly_z_threshold

    # -- source management ----------------------------------------------------
    def register_source(self, client: SourceClient) -> None:
        self.sources[client.source_name] = client

    def poll_all_sources(self) -> List[Mention]:
        """Call this on a schedule (cron / Celery beat) to pull new mentions."""
        new_mentions: List[Mention] = []
        for client in self.sources.values():
            for raw in client.fetch_new_mentions():
                new_mentions.append(self.ingest(raw))
        return new_mentions

    # -- ingestion --------------------------------------------------------------
    def ingest(self, raw: dict) -> Mention:
        mention = Mention(
            id=raw["id"],
            source=raw["source"],
            text=raw["text"],
            author=raw.get("author"),
            timestamp=raw.get("timestamp", datetime.utcnow()),
            url=raw.get("url"),
        )
        mention.nlp = self.nlp.analyze(mention.text)
        self.mentions.append(mention)
        return mention

    def ingest_batch(self, raw_list: List[dict]) -> List[Mention]:
        return [self.ingest(r) for r in raw_list]

    # -- sentiment velocity -----------------------------------------------------
    def sentiment_velocity(self, reference_time: Optional[datetime] = None,
                            bucket_minutes: int = 60) -> List[dict]:
        """
        Buckets mentions into time windows and returns, per bucket:
          - avg_polarity
          - negative_share
          - velocity = change in negative_share vs. the previous bucket
        A large positive velocity = negative sentiment accelerating fast,
        which is the actual early-warning signal (not just a low absolute score).
        """
        if not self.mentions:
            return []
        reference_time = reference_time or max(m.timestamp for m in self.mentions)
        window_start = reference_time - self.velocity_window
        relevant = sorted(
            [m for m in self.mentions if window_start <= m.timestamp <= reference_time],
            key=lambda m: m.timestamp,
        )
        if not relevant:
            return []

        buckets: Dict[datetime, List[Mention]] = {}
        for m in relevant:
            bucket_key = m.timestamp.replace(
                minute=(m.timestamp.minute // bucket_minutes) * bucket_minutes,
                second=0, microsecond=0,
            )
            buckets.setdefault(bucket_key, []).append(m)

        results = []
        prev_negative_share = None
        for bucket_time in sorted(buckets):
            group = buckets[bucket_time]
            avg_polarity = mean(m.nlp.polarity_score for m in group)
            negative_share = sum(1 for m in group if m.nlp.polarity_label == "negative") / len(group)
            velocity = 0.0 if prev_negative_share is None else negative_share - prev_negative_share
            results.append({
                "bucket_start": bucket_time,
                "mention_count": len(group),
                "avg_polarity": round(avg_polarity, 3),
                "negative_share": round(negative_share, 3),
                "sentiment_velocity": round(velocity, 3),
            })
            prev_negative_share = negative_share
        return results

    # -- anomaly detection --------------------------------------------------------
    def detect_anomalies(self, bucket_minutes: int = 60) -> List[dict]:
        """
        Flags buckets whose negative_share (or its velocity) is a statistical
        outlier (z-score beyond `anomaly_z_threshold`) vs. the rest of the window
        -- i.e. a real spike, not normal day-to-day noise.
        """
        series = self.sentiment_velocity(bucket_minutes=bucket_minutes)
        if len(series) < 3:
            return []

        neg_shares = [b["negative_share"] for b in series]
        mu, sigma = mean(neg_shares), pstdev(neg_shares) or 1e-9

        anomalies = []
        for b in series:
            z = (b["negative_share"] - mu) / sigma
            if z >= self.anomaly_z_threshold:
                anomalies.append({**b, "z_score": round(z, 2), "alert": "NEGATIVE_SENTIMENT_SPIKE"})
        return anomalies

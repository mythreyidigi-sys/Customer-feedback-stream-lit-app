"""
trust_ethics.py
================
1. BiasAuditor      -- audits the sentiment/NLP model itself for systematic
                        bias across subgroups (e.g. dialect, language variety,
                        review length) using paired/counterfactual test sets,
                        rather than trusting a black-box score.
2. TransparencyLog  -- records *why* a reputation score changed, event by
                        event, so a "transparency dashboard" can show the
                        causal chain instead of just a number moving.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean
from typing import Dict, List, Optional, Tuple

from .nlp_engine import NLPEngine


@dataclass
class BiasTestCase:
    """A pair of texts that should score the same if the model is unbiased,
    differing only in the protected attribute being tested (e.g. dialect)."""
    group_a_text: str
    group_a_label: str
    group_b_text: str
    group_b_label: str
    expected_equal: bool = True   # True: scores SHOULD match; both mean the same thing


@dataclass
class BiasAuditResult:
    group_a_label: str
    group_b_label: str
    n_pairs: int
    mean_score_gap: float          # average |score_a - score_b| across pairs
    max_score_gap: float
    flagged_pairs: List[Tuple[str, str, float]]  # (text_a, text_b, gap) over threshold


class BiasAuditor:
    """
    Runs paired-text bias audits against the NLPEngine and produces a report
    that can be attached to a "model card" / transparency dashboard, instead
    of shipping the sentiment model as an unaudited black box.
    """

    def __init__(self, nlp_engine: Optional[NLPEngine] = None, flag_threshold: float = 0.3):
        self.nlp = nlp_engine or NLPEngine()
        self.flag_threshold = flag_threshold

    def audit(self, test_cases: List[BiasTestCase]) -> Dict[str, BiasAuditResult]:
        """Groups test cases by (group_a_label, group_b_label) pair and audits each group."""
        grouped: Dict[Tuple[str, str], List[BiasTestCase]] = {}
        for tc in test_cases:
            key = (tc.group_a_label, tc.group_b_label)
            grouped.setdefault(key, []).append(tc)

        report: Dict[str, BiasAuditResult] = {}
        for (label_a, label_b), cases in grouped.items():
            gaps = []
            flagged = []
            for tc in cases:
                score_a = self.nlp.analyze(tc.group_a_text).polarity_score
                score_b = self.nlp.analyze(tc.group_b_text).polarity_score
                gap = abs(score_a - score_b)
                gaps.append(gap)
                if tc.expected_equal and gap >= self.flag_threshold:
                    flagged.append((tc.group_a_text, tc.group_b_text, round(gap, 3)))

            key_name = f"{label_a}_vs_{label_b}"
            report[key_name] = BiasAuditResult(
                group_a_label=label_a,
                group_b_label=label_b,
                n_pairs=len(cases),
                mean_score_gap=round(mean(gaps), 3),
                max_score_gap=round(max(gaps), 3),
                flagged_pairs=flagged,
            )
        return report

    @staticmethod
    def summarize(report: Dict[str, BiasAuditResult]) -> List[dict]:
        return [{
            "comparison": key,
            "pairs_tested": r.n_pairs,
            "mean_gap": r.mean_score_gap,
            "max_gap": r.max_score_gap,
            "flagged_pairs": len(r.flagged_pairs),
            "status": "REVIEW_NEEDED" if r.flagged_pairs else "OK",
        } for key, r in report.items()]


@dataclass
class TransparencyEvent:
    timestamp: datetime
    entity: str                # brand / topic / individual the score belongs to
    metric: str                # e.g. "reputation_score", "topic_risk_score"
    old_value: Optional[float]
    new_value: float
    reason: str                 # human-readable causal explanation
    contributing_mention_ids: List[str] = field(default_factory=list)


class TransparencyLog:
    """
    Append-only log of every score change and *why* it happened, so the
    transparency dashboard can answer "why did this number move?" instead of
    just showing that it did.
    """

    def __init__(self):
        self._events: List[TransparencyEvent] = []

    def record(self, entity: str, metric: str, new_value: float, reason: str,
               old_value: Optional[float] = None,
               contributing_mention_ids: Optional[List[str]] = None) -> TransparencyEvent:
        event = TransparencyEvent(
            timestamp=datetime.utcnow(),
            entity=entity,
            metric=metric,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            contributing_mention_ids=contributing_mention_ids or [],
        )
        self._events.append(event)
        return event

    def history_for(self, entity: str, metric: Optional[str] = None) -> List[TransparencyEvent]:
        return [e for e in self._events
                if e.entity == entity and (metric is None or e.metric == metric)]

    def dashboard_feed(self, limit: int = 50) -> List[dict]:
        recent = sorted(self._events, key=lambda e: e.timestamp, reverse=True)[:limit]
        return [{
            "timestamp": e.timestamp.isoformat(),
            "entity": e.entity,
            "metric": e.metric,
            "change": None if e.old_value is None else round(e.new_value - e.old_value, 3),
            "new_value": e.new_value,
            "reason": e.reason,
        } for e in recent]

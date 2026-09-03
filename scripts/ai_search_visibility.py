"""
ai_search_visibility.py
========================
Tracks how a brand/individual appears in AI-generated answers (AI Overviews,
chat-assistant answers) -- not just traditional blue-link rankings. This is
queried, not crawled: you send representative prompts to the AI surfaces you
care about and record whether/how the brand is mentioned.

`AISurfaceClient` is a stub -- point `query` at whatever you use to probe
each surface (a documented API where available, or a monitored/consented
manual-query workflow where it isn't). This module only owns the
scoring/tracking/trend logic.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional


@dataclass
class VisibilityCheck:
    surface: str            # e.g. "ai_overview", "chat_assistant_x"
    prompt: str
    timestamp: datetime
    brand_mentioned: bool
    sentiment_of_mention: Optional[str]   # positive / neutral / negative / None
    position_or_prominence: Optional[str]  # "first_mentioned" / "listed" / "not_mentioned"
    raw_answer_excerpt: str


class AISurfaceClient:
    """Stub -- replace `query` with a real call to the AI surface you're probing."""
    def __init__(self, surface_name: str, query_fn: Optional[Callable[[str], str]] = None):
        self.surface_name = surface_name
        self._query_fn = query_fn or (lambda prompt: "")

    def query(self, prompt: str) -> str:
        return self._query_fn(prompt)


class AISearchVisibilityTracker:
    """
    Runs a fixed panel of representative prompts against configured AI
    surfaces on a schedule, and tracks how visibility/sentiment trends
    over time -- the AI-answer equivalent of rank tracking.
    """

    def __init__(self, brand_name: str, tracked_prompts: List[str]):
        self.brand_name = brand_name
        self.tracked_prompts = tracked_prompts
        self.surfaces: Dict[str, AISurfaceClient] = {}
        self.history: List[VisibilityCheck] = []

    def register_surface(self, client: AISurfaceClient) -> None:
        self.surfaces[client.surface_name] = client

    def run_checks(self) -> List[VisibilityCheck]:
        results = []
        for surface_name, client in self.surfaces.items():
            for prompt in self.tracked_prompts:
                answer = client.query(prompt)
                check = self._score_answer(surface_name, prompt, answer)
                results.append(check)
                self.history.append(check)
        return results

    def _score_answer(self, surface: str, prompt: str, answer: str) -> VisibilityCheck:
        lowered = answer.lower()
        mentioned = self.brand_name.lower() in lowered

        sentiment = None
        position = "not_mentioned"
        if mentioned:
            idx = lowered.find(self.brand_name.lower())
            position = "first_mentioned" if idx < len(lowered) * 0.25 else "listed"
            negative_markers = ["complaint", "issue", "poor", "bad", "avoid", "problem"]
            positive_markers = ["recommend", "popular", "well-reviewed", "good", "best"]
            if any(w in lowered for w in negative_markers):
                sentiment = "negative"
            elif any(w in lowered for w in positive_markers):
                sentiment = "positive"
            else:
                sentiment = "neutral"

        return VisibilityCheck(
            surface=surface,
            prompt=prompt,
            timestamp=datetime.utcnow(),
            brand_mentioned=mentioned,
            sentiment_of_mention=sentiment,
            position_or_prominence=position,
            raw_answer_excerpt=answer[:280],
        )

    def visibility_rate(self, surface: Optional[str] = None) -> float:
        relevant = [c for c in self.history if surface is None or c.surface == surface]
        if not relevant:
            return 0.0
        return round(sum(1 for c in relevant if c.brand_mentioned) / len(relevant), 3)

    def trend_summary(self) -> List[dict]:
        by_surface: Dict[str, List[VisibilityCheck]] = {}
        for c in self.history:
            by_surface.setdefault(c.surface, []).append(c)
        summary = []
        for surface, checks in by_surface.items():
            mentioned = [c for c in checks if c.brand_mentioned]
            summary.append({
                "surface": surface,
                "checks_run": len(checks),
                "visibility_rate": round(len(mentioned) / len(checks), 3),
                "negative_mention_rate": round(
                    sum(1 for c in mentioned if c.sentiment_of_mention == "negative") / max(len(mentioned), 1), 3
                ),
            })
        return summary

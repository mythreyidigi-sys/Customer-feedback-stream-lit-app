"""
reputation_management
======================
A modular add-on layer for an existing online Reputation Management app.

Each module below maps directly to one feature area you listed:

    monitoring.py            -> Real-time cross-platform mention tracking + Sentiment Velocity
    nlp_engine.py             -> NLP beyond pos/neg/neutral (sarcasm, emotion, cultural-context flags)
    prediction.py             -> Predictive risk flagging + competitive benchmarking
    response.py                -> AI-assisted, human-approved response drafting
    ai_search_visibility.py   -> Tracking/influencing presence in AI Overviews / AI search answers
    digital_footprint.py      -> Personal digital-footprint audits + takedown workflow
    trust_ethics.py            -> Bias auditing of the sentiment models + transparency dashboard data

All modules are written to be plugged into an existing app:
  - They accept plain Python dicts/lists (JSON-serializable) in and out, so they drop
    straight behind a REST/GraphQL layer (Flask/FastAPI/Django) or into a task queue (Celery).
  - Any place that would call a real external API (Twitter/X, Google, NewsAPI, a data-broker
    lookup service, an LLM provider) is isolated behind a small "client" object you can swap
    for your real integration -- everything else (scoring, detection, workflow) is fully
    functional as-is.
"""

from .monitoring import MentionMonitor, Mention
from .nlp_engine import NLPEngine
from .prediction import RiskPredictor, CompetitiveBenchmark
from .response import ResponseDrafter
from .ai_search_visibility import AISearchVisibilityTracker
from .digital_footprint import DigitalFootprintAuditor
from .trust_ethics import BiasAuditor, TransparencyLog

__all__ = [
    "MentionMonitor", "Mention",
    "NLPEngine",
    "RiskPredictor", "CompetitiveBenchmark",
    "ResponseDrafter",
    "AISearchVisibilityTracker",
    "DigitalFootprintAuditor",
    "BiasAuditor", "TransparencyLog",
]

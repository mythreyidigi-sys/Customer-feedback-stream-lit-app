"""
nlp_engine.py
=============
Goes beyond positive/negative/neutral classification.

Real classifier   : VADER (vaderSentiment) gives a continuous, well-validated
                     polarity score -- swap `self._vader` for your production
                     model (e.g. a fine-tuned transformer) without touching
                     the rest of the app.
Sarcasm detection  : lightweight rule/heuristic layer (punctuation contrast,
                     positive-word + negative-context clash, exaggeration
                     markers). Good enough to *flag for human review*; wire
                     in a trained sarcasm classifier for production-grade
                     accuracy.
Emotion detection  : lexicon-based multi-label emotion tagging (anger, joy,
                     disgust, fear, sadness, trust) using NRC-style word
                     lists. Swap for a transformer emotion model if desired.
Cultural context   : simple heuristic that flags idioms/slang/region markers
                     so a human reviewer double-checks machine scoring
                     before it's trusted, instead of silently mis-scoring.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Small emotion lexicon (extend/replace with NRC Emotion Lexicon in prod)
# ---------------------------------------------------------------------------
_EMOTION_LEXICON: Dict[str, List[str]] = {
    "anger": ["furious", "angry", "outraged", "livid", "infuriating", "rude", "disrespectful"],
    "disgust": ["disgusting", "gross", "filthy", "unhygienic", "revolting", "nasty"],
    "fear": ["worried", "scared", "unsafe", "risky", "afraid", "dangerous"],
    "joy": ["delighted", "amazing", "wonderful", "love", "fantastic", "great", "happy"],
    "sadness": ["disappointed", "sad", "let down", "heartbroken", "unfortunate"],
    "trust": ["reliable", "trustworthy", "consistent", "honest", "dependable"],
    "surprise": ["shocked", "unexpected", "surprised", "unbelievable"],
}

_SARCASM_MARKERS = [
    r"\byeah\s+right\b", r"\bsure\b.{0,15}\bright\b", r"\bwow[.!]{2,}",
    r"\bgreat[.!]{2,}\s*not\b", r"\btotally\b.{0,10}\b(love|great|fine)\b.{0,10}\bnot\b",
    r"[.!]{3,}", r"\U0001F644",  # roll-eyes emoji
]

_SLANG_OR_REGIONAL_MARKERS = [
    "lol", "lmao", "meh", "y'all", "bruh", "vibe", "ngl", "fr fr",
    "not gonna lie", "no cap",
]


@dataclass
class NLPResult:
    text: str
    polarity_score: float          # -1..+1, from VADER compound score
    polarity_label: str            # negative / neutral / positive
    sarcasm_flag: bool
    sarcasm_confidence: float      # 0..1 heuristic confidence
    emotions: Dict[str, float]     # emotion -> 0..1 relative intensity
    dominant_emotion: str
    cultural_context_flag: bool    # True => recommend human review
    needs_human_review: bool = field(init=False)

    def __post_init__(self):
        # Route anything ambiguous or high-stakes to a human reviewer instead
        # of silently trusting the automated score.
        self.needs_human_review = (
            self.sarcasm_flag or self.cultural_context_flag or
            (self.polarity_label == "neutral" and self.dominant_emotion in {"anger", "disgust", "fear"})
        )


class NLPEngine:
    """Multi-signal NLP layer: polarity + sarcasm + emotion + context flags."""

    def __init__(self):
        self._vader = SentimentIntensityAnalyzer()

    # -- public API ---------------------------------------------------------
    def analyze(self, text: str) -> NLPResult:
        polarity_score = self._vader.polarity_scores(text)["compound"]
        polarity_label = self._label_from_score(polarity_score)

        sarcasm_flag, sarcasm_conf = self._detect_sarcasm(text, polarity_score)
        emotions = self._detect_emotions(text)
        dominant_emotion = max(emotions, key=emotions.get) if any(emotions.values()) else "none"
        cultural_flag = self._detect_cultural_context(text)

        return NLPResult(
            text=text,
            polarity_score=round(polarity_score, 3),
            polarity_label=polarity_label,
            sarcasm_flag=sarcasm_flag,
            sarcasm_confidence=round(sarcasm_conf, 2),
            emotions={k: round(v, 2) for k, v in emotions.items()},
            dominant_emotion=dominant_emotion,
            cultural_context_flag=cultural_flag,
        )

    def batch_analyze(self, texts: List[str]) -> List[NLPResult]:
        return [self.analyze(t) for t in texts]

    # -- internals ------------------------------------------------------------
    @staticmethod
    def _label_from_score(score: float) -> str:
        if score >= 0.05:
            return "positive"
        if score <= -0.05:
            return "negative"
        return "neutral"

    @staticmethod
    def _detect_sarcasm(text: str, polarity_score: float) -> tuple[bool, float]:
        lowered = text.lower()
        hits = sum(1 for pattern in _SARCASM_MARKERS if re.search(pattern, lowered))
        # Positive-sounding words + exclamation-heavy negative-context clash
        positive_words = {"great", "love", "amazing", "perfect", "wonderful", "best"}
        has_positive_word = any(w in lowered for w in positive_words)
        exaggerated_punct = bool(re.search(r"[!?]{2,}", text))
        clash = has_positive_word and exaggerated_punct and polarity_score < 0.3

        score = hits * 0.35 + (0.3 if clash else 0.0)
        confidence = min(score, 1.0)
        return confidence >= 0.3, confidence

    @staticmethod
    def _detect_emotions(text: str) -> Dict[str, float]:
        lowered = text.lower()
        scores = {emo: 0.0 for emo in _EMOTION_LEXICON}
        total_hits = 0
        for emo, words in _EMOTION_LEXICON.items():
            hits = sum(lowered.count(w) for w in words)
            scores[emo] = hits
            total_hits += hits
        if total_hits == 0:
            return scores
        return {emo: v / total_hits for emo, v in scores.items()}

    @staticmethod
    def _detect_cultural_context(text: str) -> bool:
        lowered = text.lower()
        return any(marker in lowered for marker in _SLANG_OR_REGIONAL_MARKERS)

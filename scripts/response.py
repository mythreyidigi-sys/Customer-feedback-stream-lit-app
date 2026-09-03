"""
response.py
===========
AI-assisted response drafting, tailored to the sentiment/tone/emotion of the
original mention, with a mandatory human-approval step before anything is
sent -- this preserves authenticity and avoids automated replies going out
on sensitive (e.g. hygiene, safety) complaints.

`LLMClient` is a thin stub: point `generate` at your real LLM call
(Anthropic API, OpenAI, Groq, etc.) -- the templating/approval-workflow logic
around it doesn't change.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, List, Optional

from .monitoring import Mention
from .nlp_engine import NLPResult


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    EDITED_AND_APPROVED = "edited_and_approved"
    REJECTED = "rejected"


@dataclass
class DraftResponse:
    mention_id: str
    draft_text: str
    tone: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    final_text: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None


class LLMClient:
    """Stub -- replace `generate` with a real LLM API call."""
    def __init__(self, generate_fn: Optional[Callable[[str], str]] = None):
        self._generate_fn = generate_fn or self._default_template_generate

    def generate(self, prompt: str) -> str:
        return self._generate_fn(prompt)

    @staticmethod
    def _default_template_generate(prompt: str) -> str:
        # Deterministic fallback so the module works with zero external deps.
        return ("Thank you for sharing this feedback -- we take it seriously and "
                "would like to make it right. Could you share more details "
                "(branch, date/time) via DM so our team can follow up directly?")


_TONE_BY_EMOTION = {
    "anger": "de-escalating and accountable",
    "disgust": "urgent, formal, and reassuring on standards",
    "fear": "reassuring and safety-focused",
    "sadness": "empathetic and warm",
    "joy": "warm and appreciative",
    "trust": "appreciative and consistent",
    "surprise": "clarifying and informative",
    "none": "neutral and professional",
}


class ResponseDrafter:
    """Drafts a reply matched to sentiment/tone, then routes it for human approval."""

    def __init__(self, llm_client: Optional[LLMClient] = None, brand_voice: str = "warm, concise, non-defensive"):
        self.llm = llm_client or LLMClient()
        self.brand_voice = brand_voice
        self._queue: List[DraftResponse] = []

    def draft(self, mention: Mention) -> DraftResponse:
        nlp: NLPResult = mention.nlp
        tone = _TONE_BY_EMOTION.get(nlp.dominant_emotion if nlp else "none", "neutral and professional")

        prompt = self._build_prompt(mention, tone)
        draft_text = self.llm.generate(prompt)

        response = DraftResponse(mention_id=mention.id, draft_text=draft_text, tone=tone)
        self._queue.append(response)
        return response

    def _build_prompt(self, mention: Mention, tone: str) -> str:
        sensitive_topics_note = ""
        if mention.nlp and mention.nlp.needs_human_review:
            sensitive_topics_note = (
                " NOTE: this mention was flagged for human review (sarcasm/cultural-context/"
                "ambiguous score) -- draft should be conservative and non-committal on specifics."
            )
        return (
            f"Brand voice: {self.brand_voice}\n"
            f"Required tone: {tone}\n"
            f"Original mention ({mention.source}): \"{mention.text}\"\n"
            f"Write a short, authentic public reply.{sensitive_topics_note}"
        )

    # -- human-in-the-loop approval workflow --------------------------------------
    def pending_for_review(self) -> List[DraftResponse]:
        return [r for r in self._queue if r.status == ApprovalStatus.PENDING]

    def approve(self, mention_id: str, reviewer: str, edited_text: Optional[str] = None) -> DraftResponse:
        response = self._find(mention_id)
        if edited_text:
            response.final_text = edited_text
            response.status = ApprovalStatus.EDITED_AND_APPROVED
        else:
            response.final_text = response.draft_text
            response.status = ApprovalStatus.APPROVED
        response.reviewed_by = reviewer
        response.reviewed_at = datetime.utcnow()
        return response

    def reject(self, mention_id: str, reviewer: str) -> DraftResponse:
        response = self._find(mention_id)
        response.status = ApprovalStatus.REJECTED
        response.reviewed_by = reviewer
        response.reviewed_at = datetime.utcnow()
        return response

    def _find(self, mention_id: str) -> DraftResponse:
        for r in self._queue:
            if r.mention_id == mention_id:
                return r
        raise KeyError(f"No draft found for mention_id={mention_id}")

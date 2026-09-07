"""
cx_common.py
------------
Shared helpers used by all four CX-analytics extension scripts:
  1. emotion_classification.py
  2. root_cause_emotion_analysis.py
  3. escalation_detection.py
  4. empathetic_reply_generator.py

Uses the Groq API (same provider already used for HDBSCAN cluster labeling
in the main pipeline) via its OpenAI-compatible chat completions endpoint.

Install:
    pip install groq pandas --break-system-packages

Auth:
    export GROQ_API_KEY="your_key_here"

If no API key is set, every function below falls back to a lightweight
keyword-based heuristic so the scripts remain runnable/testable offline.
"""

import os
import json
import time
import re

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

GROQ_MODEL = "llama-3.1-8b-instant"  # fast + cheap; swap for llama-3.3-70b-versatile for higher accuracy
MAX_RETRIES = 3
RETRY_DELAY_SEC = 2


def get_client():
    """Return a Groq client, or None if unavailable (triggers fallback heuristics)."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not _GROQ_AVAILABLE or not api_key:
        return None
    return Groq(api_key=api_key)


def call_groq_json(client, system_prompt: str, user_prompt: str) -> dict:
    """
    Call Groq chat completions expecting a strict JSON object back.
    Retries on transient failures; returns {} on final failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            return json.loads(raw)
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"[cx_common] Groq call failed after {MAX_RETRIES} attempts: {e}")
                return {}
            time.sleep(RETRY_DELAY_SEC * attempt)
    return {}


def call_groq_text(client, system_prompt: str, user_prompt: str) -> str:
    """Call Groq chat completions expecting free-text back (used for reply drafting)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.6,
                max_tokens=250,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"[cx_common] Groq call failed after {MAX_RETRIES} attempts: {e}")
                return ""
            time.sleep(RETRY_DELAY_SEC * attempt)
    return ""


def clean_review_text(text: str) -> str:
    """Light normalization before sending text to the LLM (keeps token usage down)."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1200]  # cap length; long reviews rarely need more context for these tasks

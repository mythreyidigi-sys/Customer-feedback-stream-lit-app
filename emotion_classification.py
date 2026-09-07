"""
emotion_classification.py
--------------------------
USE CASE 1: Emotion-layer classification.

Goes beyond positive/negative/neutral sentiment by classifying each review
into a fixed taxonomy of restaurant-relevant emotions. This is what
differentiates the project from Liu (2012) / Pang & Lee (2008) style
sentiment analysis: "frustration" and "betrayal" both read as "negative"
under sentiment analysis, but drive very different management responses.

INPUT  : CSV with at least a review text column (default: 'review_text')
OUTPUT : Same CSV + two new columns:
           - emotion        (single dominant emotion label)
           - emotion_conf   (LLM's confidence 0-1, or 'heuristic' flag if fallback used)

USAGE:
    python emotion_classification.py --input cleaned_reviews.csv \
                                      --output reviews_with_emotion.csv \
                                      --text-col review_text
"""

import argparse
import pandas as pd
from tqdm import tqdm

from cx_common import get_client, call_groq_json, clean_review_text

# Fixed emotion taxonomy -- kept small and business-relevant on purpose.
# A large open taxonomy (20+ emotions) fragments cluster sizes and becomes
# unusable in a Power BI slicer.
EMOTION_TAXONOMY = [
    "delight",          # genuinely positive, often praise for a specific dish/staff member
    "disappointment",   # expectation not met, but not angry
    "frustration",       # active annoyance, usually operational (wait time, mix-ups)
    "betrayal",          # felt misled -- price/portion/quality vs. expectation gap
    "relief",             # a past problem was resolved / expectations were exceeded after doubt
    "nostalgia",          # emotional attachment, "used to be better", repeat-customer sentiment
    "neutral",            # purely factual/descriptive, no strong emotional charge
]

SYSTEM_PROMPT = f"""You are an expert in customer experience analytics for restaurants.
Classify the dominant emotion expressed in a customer review into EXACTLY ONE of these categories:
{", ".join(EMOTION_TAXONOMY)}

Definitions:
- delight: genuine positive emotion, praise, enthusiasm
- disappointment: expectation not met, mild negative, resigned tone
- frustration: active annoyance about an operational failure (wait, mistakes, staff)
- betrayal: felt misled or cheated (price/portion/quality gap vs. promise)
- relief: a prior worry was resolved, or a low expectation was pleasantly exceeded
- nostalgia: comparison to a better past experience, long-time customer sentiment
- neutral: purely factual, no clear emotional charge

Respond ONLY as strict JSON: {{"emotion": "<one_of_the_categories>", "confidence": <float 0 to 1>}}
No preamble, no explanation, no markdown fences."""


# ---- Fallback heuristic (keyword-based) used when no GROQ_API_KEY is set ----
_HEURISTIC_KEYWORDS = {
    "delight": ["amazing", "loved", "excellent", "best", "fantastic", "delicious", "great experience"],
    "betrayal": ["cheated", "misleading", "overpriced", "scam", "false advertising", "not worth"],
    "frustration": ["waited", "waiting", "slow", "rude", "ignored", "unacceptable", "worst service"],
    "disappointment": ["expected better", "not up to the mark", "disappointed", "average", "mediocre"],
    "relief": ["surprised", "better than expected", "resolved", "sorted out", "pleasantly"],
    "nostalgia": ["used to be", "not like before", "earlier it was", "quality has dropped since"],
}


def heuristic_emotion(text: str) -> dict:
    text_lower = text.lower()
    for emotion, keywords in _HEURISTIC_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return {"emotion": emotion, "confidence": "heuristic"}
    return {"emotion": "neutral", "confidence": "heuristic"}


def classify_emotions(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    client = get_client()
    if client is None:
        print("[emotion_classification] No GROQ_API_KEY found -- using keyword heuristic fallback.")

    emotions, confidences = [], []
    for text in tqdm(df[text_col], desc="Classifying emotions"):
        cleaned = clean_review_text(text)
        if not cleaned:
            emotions.append("neutral")
            confidences.append("empty")
            continue

        if client is not None:
            result = call_groq_json(
                client,
                SYSTEM_PROMPT,
                f"Review:\n\"\"\"{cleaned}\"\"\"",
            )
            emotion = result.get("emotion", "").lower().strip()
            if emotion not in EMOTION_TAXONOMY:
                # LLM returned something off-taxonomy or call failed -> fallback
                fallback = heuristic_emotion(cleaned)
                emotion, conf = fallback["emotion"], fallback["confidence"]
            else:
                conf = result.get("confidence", None)
        else:
            fallback = heuristic_emotion(cleaned)
            emotion, conf = fallback["emotion"], fallback["confidence"]

        emotions.append(emotion)
        confidences.append(conf)

    df = df.copy()
    df["emotion"] = emotions
    df["emotion_conf"] = confidences
    return df


def main():
    parser = argparse.ArgumentParser(description="Emotion-layer classification for CX reviews.")
    parser.add_argument("--input", required=True, help="Path to input CSV (cleaned reviews)")
    parser.add_argument("--output", required=True, help="Path to write output CSV")
    parser.add_argument("--text-col", default="review_text", help="Column containing review text")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found. Available columns: {list(df.columns)}")

    result_df = classify_emotions(df, args.text_col)
    result_df.to_csv(args.output, index=False)

    print("\n--- Emotion distribution ---")
    print(result_df["emotion"].value_counts())
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()

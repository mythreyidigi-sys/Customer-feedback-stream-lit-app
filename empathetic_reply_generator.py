"""
empathetic_reply_generator.py
--------------------------------
USE CASE 4: Empathetic reply generator.

Auto-drafts owner/manager responses to negative reviews that acknowledge the
SPECIFIC emotion + issue detected (from emotion_classification.py +
HDBSCAN issue_cluster) instead of a generic "We're sorry for the
inconvenience" template. Intended as a draft for a human to review/edit
before posting -- not for fully automated posting.

Designed to run only on reviews below a rating threshold (default <=3) to
keep API cost down -- there's no need to draft replies for 5-star reviews.

INPUT  : CSV with 'review_text', 'emotion', 'issue_cluster', 'rating',
         optionally 'chain'
OUTPUT : CSV with an added 'draft_reply' column, plus a lightweight
         .txt digest (grouped by chain) for a manager to skim and approve.

USAGE:
    python empathetic_reply_generator.py --input reviews_with_emotion.csv \
                                          --output reviews_with_replies.csv \
                                          --digest-output reply_digest.txt \
                                          --max-rating 3
"""

import argparse
import pandas as pd
from tqdm import tqdm

from cx_common import get_client, call_groq_text, clean_review_text

SYSTEM_PROMPT = """You are drafting a restaurant manager's public reply to a negative
customer review, on behalf of the business. Requirements:
- Acknowledge the SPECIFIC emotion and issue mentioned -- do not use generic phrases
  like "we're sorry for the inconvenience" or "your feedback is valuable to us".
- Keep it under 70 words.
- Be warm, specific, and non-defensive. Do not make promises the business can't verify
  (e.g. don't promise a refund or discount -- offer to take the conversation offline instead).
- Sign off as "- Team [Restaurant]" using a generic placeholder if the chain name isn't given.
- Do not use exclamation marks excessively or sound like a form letter.
Output ONLY the reply text. No preamble, no quotes, no markdown."""


# Fallback templates keyed by emotion, used when no GROQ_API_KEY is available.
# Still issue-aware via simple string substitution, though less specific than an LLM draft.
_FALLBACK_TEMPLATES = {
    "betrayal": "We hear you, and a gap between what we promised on {issue} and what you experienced "
                "isn't okay. We'd like to make this right -- please DM us your visit details. "
                "- Team {chain}",
    "frustration": "The frustration around {issue} during your visit is completely fair, and it's not "
                   "the experience we want for you. We're looking into what went wrong -- please reach "
                   "out so we can follow up directly. - Team {chain}",
    "disappointment": "Sorry we fell short on {issue} this time -- that's not the standard we hold "
                      "ourselves to. We'd appreciate the chance to do better on your next visit. "
                      "- Team {chain}",
    "nostalgia": "We understand it's frustrating when {issue} doesn't feel like it used to -- we're "
                "working on getting back to that standard. Thank you for staying with us this long. "
                "- Team {chain}",
}
_DEFAULT_TEMPLATE = "Thank you for the honest feedback on {issue}. We're taking this seriously and " \
                    "would like to understand more -- please reach out directly. - Team {chain}"


def fallback_reply(issue: str, emotion: str, chain: str) -> str:
    template = _FALLBACK_TEMPLATES.get(emotion, _DEFAULT_TEMPLATE)
    return template.format(issue=issue.lower() if isinstance(issue, str) else "your experience",
                            chain=chain if isinstance(chain, str) and chain else "our restaurant")


def generate_replies(df: pd.DataFrame, text_col: str, emotion_col: str,
                      issue_col: str, chain_col: str) -> pd.DataFrame:
    client = get_client()
    if client is None:
        print("[empathetic_reply_generator] No GROQ_API_KEY found -- using template fallback.")

    replies = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Drafting replies"):
        review_text = clean_review_text(row.get(text_col, ""))
        emotion = row.get(emotion_col, "neutral")
        issue = row.get(issue_col, "your experience") if issue_col in df.columns else "your experience"
        chain = row.get(chain_col, "") if chain_col in df.columns else ""

        if not review_text:
            replies.append("")
            continue

        if client is not None:
            user_prompt = (
                f"Review: \"\"\"{review_text}\"\"\"\n"
                f"Detected emotion: {emotion}\n"
                f"Detected issue category: {issue}\n"
                f"Restaurant chain name (use if provided, else keep generic): {chain}"
            )
            reply = call_groq_text(client, SYSTEM_PROMPT, user_prompt)
            if not reply:
                reply = fallback_reply(issue, emotion, chain)
        else:
            reply = fallback_reply(issue, emotion, chain)

        replies.append(reply)

    df = df.copy()
    df["draft_reply"] = replies
    return df


def write_digest(df: pd.DataFrame, chain_col: str, path: str):
    """Human-readable digest a manager can skim before approving replies."""
    lines = ["EMPATHETIC REPLY DIGEST", "=" * 60, ""]
    group_col = chain_col if chain_col in df.columns else None
    groups = df.groupby(group_col) if group_col else [("All reviews", df)]

    for name, group in groups:
        lines.append(f"\n### {name} ({len(group)} flagged reviews) ###\n")
        for _, row in group.iterrows():
            lines.append(f"Rating: {row.get('rating', 'NA')} | Emotion: {row.get('emotion', 'NA')} "
                         f"| Issue: {row.get('issue_cluster', 'NA')}")
            lines.append(f"Review : {str(row.get('review_text', ''))[:200]}")
            lines.append(f"Draft  : {row.get('draft_reply', '')}")
            lines.append("-" * 60)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Empathetic reply generator for negative CX reviews.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--digest-output", required=True)
    parser.add_argument("--text-col", default="review_text")
    parser.add_argument("--emotion-col", default="emotion")
    parser.add_argument("--issue-col", default="issue_cluster")
    parser.add_argument("--chain-col", default="chain")
    parser.add_argument("--rating-col", default="rating")
    parser.add_argument("--max-rating", type=int, default=3,
                        help="Only draft replies for reviews with rating <= this value")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found. Available: {list(df.columns)}")

    if args.rating_col in df.columns:
        target_df = df[df[args.rating_col] <= args.max_rating].copy()
        print(f"Drafting replies for {len(target_df)} of {len(df)} reviews "
              f"(rating <= {args.max_rating}).")
    else:
        target_df = df.copy()
        print(f"No rating column found -- drafting replies for all {len(df)} reviews.")

    result = generate_replies(target_df, args.text_col, args.emotion_col,
                               args.issue_col, args.chain_col)
    result.to_csv(args.output, index=False)
    write_digest(result, args.chain_col, args.digest_output)

    print(f"\nSaved:\n  {args.output}\n  {args.digest_output}")


if __name__ == "__main__":
    main()

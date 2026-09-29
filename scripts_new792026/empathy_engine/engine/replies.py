"""Empathetic replies and summaries: Groq LLM when GROQ_API_KEY is set, otherwise templates."""
from .config import GROQ_API_KEY, GROQ_MODEL, POSITIVE_ISSUE
from .playbook import recommend

_client = None


def groq_available() -> bool:
    return bool(GROQ_API_KEY)


def _groq(prompt: str, max_tokens=300, temperature=0.4) -> str | None:
    global _client
    if not GROQ_API_KEY:
        return None
    try:
        if _client is None:
            from groq import Groq
            _client = Groq(api_key=GROQ_API_KEY)
        r = _client.chat.completions.create(
            model=GROQ_MODEL, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content.strip()
    except Exception as e:  # network / quota problems -> fall back to templates
        print("Groq error:", e)
        return None


OPENERS = {
    "anger": "We're truly sorry — this is not the experience you should have had with us",
    "frustration": "We're sorry for the wait and the trouble this caused you",
    "disappointment": "We're sorry we let you down this time",
    "neutral": "Thank you for taking the time to share this",
}
PROMISES = {
    "Poor Experience / Hygiene Complaints": "We've arranged a full hygiene check at our {branch} branch this week.",
    "Food Quantity & Value for Money": "We're reviewing portion sizes at {branch} so every plate matches our standard.",
    "Slow Service & Staff Negligence": "We're adding staff at {branch} during peak hours so no guest waits unattended.",
    "Service Quality": "We've shared your feedback with the {branch} team and are tightening our order checks.",
    "Food Quality & Branch Consistency": "Our kitchen head is checking the recipe and freshness standards at {branch}.",
    "Pricing & Staff Performance": "The way you were treated is not acceptable; the {branch} manager is addressing it with the team.",
    "Ambience & Location": "We're looking into the seating and comfort at {branch}.",
    "Brand Reputation & Customer Satisfaction": "Your loyalty means a lot, and our {branch} manager would like to make this right.",
    "Food Variety & Fast Service": "We're making sure the {branch} menu matches what we list online.",
}


def template_reply(restaurant, branch, issue, emotion, text="") -> str:
    branch = branch or "our"
    if issue == POSITIVE_ISSUE:
        return (f"Thank you so much for your kind words! We're delighted you enjoyed your visit to {restaurant} "
                f"{branch}. We've shared your feedback with the team and look forward to serving you again.")
    opener = OPENERS.get(emotion, OPENERS["neutral"])
    promise = PROMISES.get(issue, "We've shared your feedback with the {branch} team.").format(branch=branch)
    return (f"{opener}, and thank you for telling us. {promise} We'd love the chance to welcome you back — "
            f"please reach us at feedback@{restaurant.lower().replace(' ', '')}.in so we can make it right. "
            f"— Team {restaurant}")


def draft_reply(restaurant, branch, issue, emotion, text, rating=None) -> tuple[str, str]:
    """Returns (reply, engine) where engine is 'groq' or 'template'."""
    fix = recommend(issue, branch)["action"]
    prompt = f"""You are the customer-care manager of {restaurant}, a vegetarian restaurant chain in Chennai.
Write a public reply to this review from the {branch} branch.

Review ({rating if rating else '?'}/5): "{text}"
Issue category: {issue}
Customer emotion: {emotion}
Corrective action we are taking: {fix}

Rules: 50-90 words. Warm, specific to what they said, no excuses, no discounts or promises you cannot keep,
acknowledge the emotion, mention the concrete action, invite them back. Sign off as "Team {restaurant}".
Return only the reply text."""
    out = _groq(prompt)
    return (out, "groq") if out else (template_reply(restaurant, branch, issue, emotion, text), "template")


def monthly_summary(restaurant, month_label, facts: dict) -> tuple[str, str]:
    prompt = f"""Write a 3-4 sentence executive summary for the {month_label} customer-review report of
{restaurant} (Chennai restaurant chain), for the restaurant manager. Use only these facts:
{facts}
Say what changed, the main driver, what improved, and the biggest remaining complaint. Plain English, no bullet points."""
    out = _groq(prompt, max_tokens=220, temperature=0.3)
    if out:
        return out, "groq"
    s = facts
    direction = "rose" if (s.get("score_delta") or 0) > 0 else "dipped" if (s.get("score_delta") or 0) < 0 else "held steady"
    txt = (f"{restaurant} received {s['reviews']} reviews in {month_label} and its reputation score {direction} "
           f"to {s['score']}/100 ({s['score_delta']:+.1f} pts vs last month). " if s.get("score_delta") is not None else
           f"{restaurant} received {s['reviews']} reviews in {month_label} with a reputation score of {s['score']}/100. ")
    if s.get("top_issue"):
        txt += f"{s['top_issue']} was the largest complaint theme with {s['top_issue_count']} mentions. "
    if s.get("biggest_rise"):
        txt += f"The sharpest increase was in {s['biggest_rise']}. "
    if s.get("biggest_fall"):
        txt += f"{s['biggest_fall']} improved compared with last month. "
    if s.get("spike"):
        txt += f"Watch-out: {s['spike']}."
    return txt.strip(), "template"

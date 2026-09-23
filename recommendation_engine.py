"""
recommendation_engine.py
------------------------
Recommendation Engine for the Restaurant ORM project (BITS Pilani MBA-WILP, 2024MB22535).

Three layers:
  1. PLAYBOOK      - rule-based mapping: issue category -> corrective actions (owner, timeframe, KPI)
  2. HOTSPOTS      - finds which chain x issue combinations are over-represented
                     (uses week-over-week spikes if a date column exists, otherwise
                      compares each chain's issue share against the all-chain baseline)
  3. LLM TAILORING - optional Groq call that adapts the playbook action to one specific review
  4. BANDIT (PoC)  - Thompson-sampling contextual bandit trained on a SIMULATED environment.
                     Rewards are assumption-based, NOT observed outcomes. It demonstrates how
                     the engine would learn once real intervention logs become available.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. PLAYBOOK
# ---------------------------------------------------------------------------
# Each action: id, action text, owner, timeframe, KPI to track.
PLAYBOOK: dict[str, list[dict]] = {
    "Food Quantity & Value for Money": [
        {"id": "FQ1", "action": "Standardise portion sizes with measured ladles/scoops and plating photos at every counter", "owner": "Kitchen Head", "timeframe": "1 week", "kpi": "Share of value-for-money complaints"},
        {"id": "FQ2", "action": "Introduce a value combo / thali tier for price-sensitive guests", "owner": "Marketing", "timeframe": "2-3 weeks", "kpi": "Combo uptake, avg bill, value complaints"},
        {"id": "FQ3", "action": "Audit delivery packaging so portions match dine-in quantity", "owner": "Delivery Ops", "timeframe": "1 week", "kpi": "Delivery-platform value complaints"},
        {"id": "FQ4", "action": "Communicate portion size (grams/pieces) on menu and aggregator listings", "owner": "Marketing", "timeframe": "1 week", "kpi": "Expectation-mismatch complaints"},
    ],
    "Slow Service & Staff Negligence": [
        {"id": "SS1", "action": "Add floor staff / re-roster for peak hours (12-2 pm, 7-9 pm, weekends)", "owner": "Branch Manager", "timeframe": "1 week", "kpi": "Wait-time complaints at peak"},
        {"id": "SS2", "action": "Set a service SLA (e.g., order taken within 5 min) with a table-timer check", "owner": "Branch Manager", "timeframe": "2 weeks", "kpi": "Avg order-to-serve time"},
        {"id": "SS3", "action": "Run a short attentiveness refresher training for service staff", "owner": "HR / Training", "timeframe": "2 weeks", "kpi": "Staff-negligence mentions"},
        {"id": "SS4", "action": "Introduce QR / token ordering to reduce waiting for a server", "owner": "IT / Ops", "timeframe": "3-4 weeks", "kpi": "Wait-time complaints"},
    ],
    "Service Quality": [
        {"id": "SQ1", "action": "Daily 10-minute pre-shift briefing on service standards", "owner": "Branch Manager", "timeframe": "1 week", "kpi": "Service-quality complaint share"},
        {"id": "SQ2", "action": "Mystery-diner audit of weekend service", "owner": "Quality Team", "timeframe": "2 weeks", "kpi": "Audit score"},
        {"id": "SQ3", "action": "Table-side feedback card / QR with manager callback for low scores", "owner": "Marketing", "timeframe": "2 weeks", "kpi": "Issues resolved before online posting"},
    ],
    "Food Quality & Branch Consistency": [
        {"id": "FC1", "action": "Centralise masala/batter preparation or enforce standard recipe cards", "owner": "Central Kitchen", "timeframe": "3-4 weeks", "kpi": "Consistency complaints by branch"},
        {"id": "FC2", "action": "Weekly taste audit comparing flagged branch with best-rated branch", "owner": "Quality Team", "timeframe": "1 week", "kpi": "Branch taste-audit score"},
        {"id": "FC3", "action": "Freshness rule: discard/re-cook items held beyond set time", "owner": "Kitchen Head", "timeframe": "1 week", "kpi": "Stale-food mentions"},
    ],
    "Pricing & Staff Performance": [
        {"id": "PS1", "action": "Link price increases to visible service upgrades and communicate them", "owner": "Marketing", "timeframe": "2 weeks", "kpi": "Price-related complaints"},
        {"id": "PS2", "action": "Staff incentive tied to branch review rating", "owner": "HR", "timeframe": "3-4 weeks", "kpi": "Staff-performance mentions"},
        {"id": "PS3", "action": "Benchmark prices against competing chains for top-10 items", "owner": "Finance", "timeframe": "2 weeks", "kpi": "Price perception"},
    ],
    "Ambience & Location": [
        {"id": "AL1", "action": "Re-layout seating to reduce crowding; add waiting area", "owner": "Branch Manager", "timeframe": "3-4 weeks", "kpi": "Ambience/seating complaints"},
        {"id": "AL2", "action": "Publish parking guidance / partner parking on Google & Zomato listings", "owner": "Marketing", "timeframe": "1 week", "kpi": "Parking/access complaints"},
        {"id": "AL3", "action": "Lighting, ventilation and noise check at flagged outlets", "owner": "Facilities", "timeframe": "2 weeks", "kpi": "Ambience mentions"},
    ],
    "Brand Reputation & Customer Satisfaction": [
        {"id": "BR1", "action": "Respond publicly to every negative review within 24 h (empathetic reply generator)", "owner": "ORM Team", "timeframe": "Immediate", "kpi": "Response rate, response time"},
        {"id": "BR2", "action": "Win-back offer for customers expressing 'used to be my favourite' sentiment", "owner": "Marketing / CRM", "timeframe": "2 weeks", "kpi": "Repeat-visit mentions"},
        {"id": "BR3", "action": "Monthly brand-health report to leadership from dashboard", "owner": "ORM Team", "timeframe": "Monthly", "kpi": "Reputation score trend"},
        {"id": "BR4", "action": "Amplify positive reviews (UGC) on social channels", "owner": "Marketing", "timeframe": "Ongoing", "kpi": "Positive share of voice"},
    ],
    "Poor Experience / Hygiene Complaints": [
        {"id": "HY1", "action": "Immediate hygiene audit at flagged branch (tables, cutlery, restrooms, kitchen)", "owner": "Quality Team", "timeframe": "48 hours", "kpi": "Hygiene complaints"},
        {"id": "HY2", "action": "Table-reset checklist before seating each guest", "owner": "Branch Manager", "timeframe": "1 week", "kpi": "Cleanliness mentions"},
        {"id": "HY3", "action": "Pest-control schedule review and documentation", "owner": "Facilities", "timeframe": "1 week", "kpi": "Pest/insect mentions"},
    ],
    "Food Variety & Fast Service": [
        {"id": "FV1", "action": "Sync online menu with actual evening availability", "owner": "IT / Ops", "timeframe": "1 week", "kpi": "'Not available' complaints"},
        {"id": "FV2", "action": "Add 2-3 evening specials to widen variety", "owner": "Kitchen Head", "timeframe": "3 weeks", "kpi": "Variety mentions"},
    ],
    "Overall Positive Experience": [
        {"id": "OP1", "action": "Thank reviewer and invite them to loyalty programme", "owner": "ORM Team", "timeframe": "Immediate", "kpi": "Loyalty sign-ups"},
        {"id": "OP2", "action": "Use positive themes as marketing proof points", "owner": "Marketing", "timeframe": "Ongoing", "kpi": "Campaign engagement"},
    ],
}

# The ORM app uses a finer taxonomy in places. Map those labels onto the playbook keys.
ALIASES = {
    "Staff Courtesy & Behaviour": "Slow Service & Staff Negligence",
    "Cleanliness & Restroom Hygiene": "Poor Experience / Hygiene Complaints",
    "Poor Experience & Food Hygiene Complaints": "Poor Experience / Hygiene Complaints",
    "Ambience & Seating": "Ambience & Location",
    "Parking & Accessibility": "Ambience & Location",
    "Billing & Online Ordering Issues": "Pricing & Staff Performance",
}

NEGATIVE_EMOTIONS = {"frustration": 3, "betrayal": 3, "disappointment": 2, "neutral": 1,
                     "nostalgia": 1, "relief": 0, "delight": 0}


def canonical_issue(label: str) -> str:
    return ALIASES.get(label, label)


def severity(emotion: str, urgency_score: float | None = None) -> str:
    """Return 'high' / 'medium' / 'low' from emotion and optional urgency score (0-10)."""
    e = NEGATIVE_EMOTIONS.get(str(emotion).lower(), 1)
    u = 0 if urgency_score is None or pd.isna(urgency_score) else float(urgency_score)
    score = e + u / 3
    return "high" if score >= 4.5 else "medium" if score >= 2 else "low"


def recommend(issue: str, emotion: str = "neutral", urgency_score: float | None = None,
              top_k: int = 2) -> dict:
    """Rule-based recommendation for a single review / issue."""
    key = canonical_issue(issue)
    actions = PLAYBOOK.get(key, [])
    sev = severity(emotion, urgency_score)
    # High severity: put fast-acting actions first
    fast = {"Immediate": 0, "48 hours": 1, "1 week": 2}
    ordered = sorted(actions, key=lambda a: fast.get(a["timeframe"], 5)) if sev == "high" else actions
    return {"issue": key, "severity": sev, "escalate": sev == "high",
            "actions": ordered[:top_k]}


# ---------------------------------------------------------------------------
# 2. HOTSPOTS
# ---------------------------------------------------------------------------
def find_hotspots(df: pd.DataFrame, chain_col="restaurant", issue_col="issue_cluster",
                  date_col: str | None = None, min_reviews=15, top_n=10) -> pd.DataFrame:
    """
    If date_col is given and populated: week-over-week spike (last week vs mean of previous 4).
    Otherwise: 'lift' = chain's share of an issue / all-chain share of that issue.
    lift > 1.3 means the chain has 30 % more of that issue than the market average.
    """
    d = df.copy()
    d["issue_std"] = d[issue_col].map(canonical_issue)

    if date_col and date_col in d and d[date_col].notna().mean() > 0.8:
        d["week"] = pd.to_datetime(d[date_col]).dt.to_period("W")
        wk = d.groupby([chain_col, "issue_std", "week"]).size().rename("n").reset_index()
        last = wk["week"].max()
        cur = wk[wk.week == last].set_index([chain_col, "issue_std"])["n"]
        base = (wk[(wk.week < last) & (wk.week >= last - 4)]
                .groupby([chain_col, "issue_std"])["n"].mean())
        out = pd.concat([cur.rename("this_week"), base.rename("baseline")], axis=1).fillna(0)
        out["lift"] = (out.this_week + 1) / (out.baseline + 1)
        out["method"] = "week-over-week spike"
    else:
        tab = pd.crosstab(d[chain_col], d["issue_std"])
        chain_share = tab.div(tab.sum(axis=1), axis=0)
        market_share = tab.sum() / tab.values.sum()
        lift = chain_share.div(market_share, axis=1)
        out = pd.DataFrame({"reviews": tab.stack(), "chain_share": chain_share.stack(),
                            "market_share": pd.Series(market_share).reindex(
                                tab.stack().index.get_level_values(1)).values,
                            "lift": lift.stack()})
        out = out[out.reviews >= min_reviews]
        out["method"] = "chain vs market share"

    out = out.reset_index().rename(columns={"level_0": chain_col, "level_1": "issue_std"})
    out = out[out["issue_std"] != "Overall Positive Experience"]
    out = out.sort_values("lift", ascending=False).head(top_n)
    out["recommended_action"] = out["issue_std"].map(
        lambda i: PLAYBOOK.get(i, [{"action": "-"}])[0]["action"])
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. LLM TAILORING (optional, Groq)
# ---------------------------------------------------------------------------
def tailor_with_llm(review_text: str, chain: str, rec: dict,
                    model: str = "llama-3.3-70b-versatile") -> str:
    """Adapt the playbook actions to one review. Falls back to rule text if Groq unavailable."""
    fallback = "\n".join(f"- {a['action']} (Owner: {a['owner']}, {a['timeframe']})"
                         for a in rec["actions"])
    try:
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        prompt = (
            f"You are an operations consultant for {chain}, a Chennai vegetarian restaurant chain.\n"
            f"Customer review: \"{review_text}\"\n"
            f"Issue category: {rec['issue']} | Severity: {rec['severity']}\n"
            f"Approved playbook actions:\n{fallback}\n\n"
            "Using ONLY the approved actions, write 2-3 specific, practical steps for the branch "
            "manager, each with owner and deadline. Do not invent new policies. Max 100 words."
        )
        resp = client.chat.completions.create(
            model=model, temperature=0.3,
            messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content.strip()
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# 4. CONTEXTUAL BANDIT - PROOF OF CONCEPT ON A SIMULATED ENVIRONMENT
# ---------------------------------------------------------------------------
class SimulatedRestaurantEnv:
    """
    SIMULATED environment. For each (issue, severity) context and action, a hidden
    'true' probability that the action reduces the complaint next week.
    These probabilities are ASSUMPTIONS for demonstration only, not measured data.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.true_p = {}
        for issue, actions in PLAYBOOK.items():
            if issue == "Overall Positive Experience":
                continue
            for sev in ("low", "medium", "high"):
                # Assumption: fast actions work better on high severity,
                # structural (slower) actions work better on low/medium recurring issues.
                for a in actions:
                    fast = a["timeframe"] in ("Immediate", "48 hours", "1 week")
                    base = self.rng.uniform(0.25, 0.60)
                    bonus = 0.20 if (fast and sev == "high") or (not fast and sev != "high") else 0.0
                    self.true_p[(issue, sev, a["id"])] = min(base + bonus, 0.9)
        self.contexts = sorted({(i, s) for i, s, _ in self.true_p})

    def sample_context(self):
        return self.contexts[self.rng.integers(len(self.contexts))]

    def step(self, context, action_id) -> int:
        return int(self.rng.random() < self.true_p[(*context, action_id)])

    def best_p(self, context):
        return max(p for (i, s, _), p in self.true_p.items() if (i, s) == context)


class ThompsonBandit:
    """Beta-Bernoulli Thompson sampling, one arm set per (issue, severity) context."""

    def __init__(self, seed: int = 0):
        self.alpha, self.beta = {}, {}
        self.rng = np.random.default_rng(seed)

    def choose(self, context) -> str:
        actions = [a["id"] for a in PLAYBOOK[context[0]]]
        samples = [self.rng.beta(self.alpha.get((context, a), 1), self.beta.get((context, a), 1))
                   for a in actions]
        return actions[int(np.argmax(samples))]

    def update(self, context, action_id, reward):
        k = (context, action_id)
        self.alpha[k] = self.alpha.get(k, 1) + reward
        self.beta[k] = self.beta.get(k, 1) + (1 - reward)

    def learned_policy(self) -> pd.DataFrame:
        rows = []
        for (ctx, a), al in self.alpha.items():
            be = self.beta[(ctx, a)]
            rows.append({"issue": ctx[0], "severity": ctx[1], "action_id": a,
                         "est_success": al / (al + be), "trials": al + be - 2})
        df = pd.DataFrame(rows)
        return (df.sort_values("est_success", ascending=False)
                  .groupby(["issue", "severity"]).head(1).reset_index(drop=True))


def run_simulation(rounds: int = 5000, seed: int = 42) -> dict:
    """Compare Thompson bandit vs random vs fixed rule (always first playbook action)."""
    env = SimulatedRestaurantEnv(seed)
    bandit = ThompsonBandit(seed)
    rng = np.random.default_rng(seed + 1)
    rewards = {"Bandit (Thompson)": [], "Random action": [], "Fixed rule": []}
    regret = []
    for _ in range(rounds):
        ctx = env.sample_context()
        acts = [a["id"] for a in PLAYBOOK[ctx[0]]]
        a_b = bandit.choose(ctx)
        r = env.step(ctx, a_b)
        bandit.update(ctx, a_b, r)
        rewards["Bandit (Thompson)"].append(r)
        rewards["Random action"].append(env.step(ctx, acts[rng.integers(len(acts))]))
        rewards["Fixed rule"].append(env.step(ctx, acts[0]))
        regret.append(env.best_p(ctx) - env.true_p[(*ctx, a_b)])
    return {"rewards": pd.DataFrame(rewards), "regret": np.cumsum(regret),
            "policy": bandit.learned_policy(), "env": env}

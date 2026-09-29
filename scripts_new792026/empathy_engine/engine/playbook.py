"""Playbook: the recommended fix, owner and deadline for each issue category. Edit freely."""

PLAYBOOK = {
    "Poor Experience / Hygiene Complaints": {
        "action": "Immediate hygiene audit at {branch}; table-reset and pest-control checklist every shift",
        "owner": "Quality team", "sla_hours": 48,
        "expected": "Hygiene complaints back to 4-week average within 2 weeks",
    },
    "Food Quantity & Value for Money": {
        "action": "Standardise portion sizes at {branch} with weighed ladles and a plating chart; review combo pricing",
        "owner": "Kitchen head", "sla_hours": 168,
        "expected": "Value-for-money complaints down 20% next month",
    },
    "Slow Service & Staff Negligence": {
        "action": "Re-roster peak hours at {branch}; floor manager to check every table within 5 minutes",
        "owner": "Branch manager", "sla_hours": 72,
        "expected": "Average wait under 15 minutes at peak",
    },
    "Service Quality": {
        "action": "Order-accuracy check at the pass and bill verification at {branch}; 15-min refresher for servers",
        "owner": "Branch manager", "sla_hours": 72,
        "expected": "Wrong-order / billing complaints down 25%",
    },
    "Food Quality & Branch Consistency": {
        "action": "Taste audit at {branch} against the central recipe card; check holding temperatures",
        "owner": "Kitchen head", "sla_hours": 96,
        "expected": "Consistent taste scores across branches",
    },
    "Pricing & Staff Performance": {
        "action": "Courtesy coaching for the {branch} team; manager to call the guest back personally",
        "owner": "Operations head", "sla_hours": 72,
        "expected": "Staff-behaviour complaints down 30%",
    },
    "Ambience & Location": {
        "action": "Check seating layout, AC and parking signage at {branch}; add a valet or parking tie-up if needed",
        "owner": "Operations head", "sla_hours": 336,
        "expected": "Ambience complaints stable or falling",
    },
    "Brand Reputation & Customer Satisfaction": {
        "action": "Personal call-back from the {branch} manager with a return-visit invitation",
        "owner": "ORM team", "sla_hours": 24,
        "expected": "Win back lapsed regulars; lift reputation score",
    },
    "Food Variety & Fast Service": {
        "action": "Align the {branch} menu with what is listed online; flag sold-out items on aggregator apps",
        "owner": "Branch manager", "sla_hours": 168,
        "expected": "Fewer 'not available' complaints",
    },
    "Overall Positive Experience": {
        "action": "Thank the guest; share praise with the {branch} team",
        "owner": "ORM team", "sla_hours": 72,
        "expected": "Keep reply rate above 90%",
    },
}
DEFAULT = {"action": "Review the complaint with the {branch} manager and agree a fix", "owner": "Branch manager",
           "sla_hours": 72, "expected": "Issue closed within SLA"}


def recommend(issue: str, branch: str = "the branch") -> dict:
    p = PLAYBOOK.get(issue, DEFAULT)
    return {**p, "action": p["action"].format(branch=branch or "the branch")}


def deadline_text(hours: int) -> str:
    return f"within {hours} hours" if hours <= 72 else f"within {hours // 24} days"

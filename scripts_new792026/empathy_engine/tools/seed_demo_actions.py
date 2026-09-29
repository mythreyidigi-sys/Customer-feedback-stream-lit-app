"""SAMPLE DATA ONLY: pretend the team already handled most older escalations, so the Action center
and Monthly report KPIs (reply rate, reply time, resolved) show realistic numbers.
Do NOT run this on your real data.

    python tools/seed_demo_actions.py
"""
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import db  # noqa: E402
from engine.config import HIGH_THRESHOLD, TEAM  # noqa: E402
from engine.playbook import recommend  # noqa: E402
from engine.replies import template_reply  # noqa: E402

random.seed(3)
df = db.load_reviews()
as_of = df["review_date"].max()
neg = df[(df["sentiment"] == "negative") & (df["review_date"] < as_of - pd.Timedelta(days=5))]
n = 0
for r in neg.itertuples():
    if random.random() > 0.85:
        continue
    replied = r.review_date + pd.Timedelta(hours=random.randint(3, 40))
    fields = dict(status="Replied", assignee=random.choice(TEAM[1:]),
                  reply_text=template_reply(r.restaurant, r.branch, r.issue, r.emotion),
                  replied_at=replied.isoformat(timespec="seconds"),
                  action_text=recommend(r.issue, r.branch)["action"])
    if r.urgency >= HIGH_THRESHOLD and random.random() < 0.8:
        fields.update(status="Resolved", resolved_at=(replied + pd.Timedelta(days=2)).isoformat(timespec="seconds"))
    db.upsert_action(int(r.id), **fields)
    n += 1
print(f"Seeded {n} demo actions.")

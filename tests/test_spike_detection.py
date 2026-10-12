"""Normal-period and spike-period tests for spike_detection (run: python -m unittest discover tests)."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spike_detection as sd  # noqa: E402

AS_OF = pd.Timestamp("2026-03-29")


def make(weekly: list[int], issue="Service Quality", branch="B1", restaurant="R1", sentiment="negative"):
    """weekly[0] = latest week; reviews placed in the middle of each 7-day bin ending AS_OF."""
    rows = []
    for k, n in enumerate(weekly):
        d = AS_OF - pd.Timedelta(days=7 * k + 3)
        rows += [{"restaurant": restaurant, "branch": branch, "issue": issue, "sentiment": sentiment,
                  "review_date": d.strftime("%Y-%m-%d")}] * n
    return pd.DataFrame(rows)


class SpikeTests(unittest.TestCase):
    def run_week(self, weekly, **kw):
        df = make(weekly, **kw)
        counts, n = sd.weekly_counts(df, AS_OF, "issue")
        return sd.evaluate(counts, n, 0).iloc[0]

    def test_normal_period_no_alert(self):
        r = self.run_week([2, 2, 1, 2, 2, 1, 2, 2, 1, 2])
        self.assertFalse(r.threshold_alert)
        self.assertFalse(r.z_alert)

    def test_spike_triggers_both(self):
        r = self.run_week([9, 2, 1, 2, 2, 1, 2, 2, 1, 2])
        self.assertTrue(r.threshold_alert)
        self.assertTrue(r.z_alert)
        self.assertGreater(r.z_score, sd.Z_THRESHOLD)
        self.assertIn("Z =", r.reason)

    def test_min_count_blocks_small_spike(self):
        r = self.run_week([2, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        self.assertFalse(r.threshold_alert)
        self.assertFalse(r.z_alert)

    def test_threshold_baseline_floor(self):
        # 3 reviews vs a zero baseline: ratio uses baseline floor of 1 -> 3x, so it alerts
        r = self.run_week([3, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        self.assertTrue(r.threshold_alert)
        self.assertAlmostEqual(r.threshold_ratio, 3.0)

    def test_short_history_marks_z_pending(self):
        r = self.run_week([5, 1, 1])
        self.assertTrue(r.threshold_alert)
        self.assertFalse(r.z_alert)
        self.assertTrue(pd.isna(r.z_score))
        self.assertTrue(r.z_status.startswith("pending"))

    def test_positive_and_non_negative_reviews_ignored(self):
        self.assertTrue(sd.complaint_rows(make([9] * 3, sentiment="positive")).empty)
        self.assertTrue(sd.complaint_rows(make([9] * 3, issue=sd.POSITIVE_ISSUE)).empty)

    def test_weeks_before_first_review_not_zero_filled(self):
        df = make([4, 1, 1, 1])
        counts, n = sd.weekly_counts(df, AS_OF, "issue")
        self.assertEqual(n, 4)

    def test_current_alerts_agreement_label(self):
        df = make([9, 2, 1, 2, 2, 1, 2, 2, 1, 2])
        a = sd.current_alerts(df, AS_OF, "issue")
        self.assertEqual(list(a["agreement"]), ["Both rules"])

    def test_empty_input(self):
        self.assertTrue(sd.current_alerts(make([]).reindex(columns=["restaurant", "branch", "issue", "sentiment",
                                                                      "review_date"]), AS_OF).empty)


if __name__ == "__main__":
    unittest.main()

"""Regression tests for malformed branch names in monthly reports."""
import sys
import unittest
from pathlib import Path

import pandas as pd

ENGINE_DIR = (Path(__file__).resolve().parents[1] / "scripts_new792026"
              / "empathy_engine (5)" / "empathy_engine")
sys.path.insert(0, str(ENGINE_DIR))

from engine.analytics import branch_table  # noqa: E402
from engine.importer import clean_branch  # noqa: E402


class BranchReportTests(unittest.TestCase):
    def test_tripadvisor_page_slug_is_not_used_as_branch(self):
        branch = (
            "Restaurant_Review-g304556-d27106440-Reviews-Geetham_Veg_Restaurant-"
            "Chennai_Madras_Chennai_District_Tamil_Nadu"
        )

        self.assertEqual(clean_branch(branch), "Main")

    def test_branch_report_cleans_existing_page_slugs_without_dropping_reviews(self):
        page_slug = (
            "Restaurant_Review-g304556-d27106440-Reviews-Geetham_Veg_Restaurant-"
            "Chennai_Madras_Chennai_District_Tamil_Nadu"
        )
        reviews = pd.DataFrame({
            "branch": [page_slug, page_slug, "Koyambedu"],
            "rating": [4.0, 3.0, 5.0],
            "sentiment": ["negative", "neutral", "positive"],
            "issue": ["Service Quality", "Service Quality", "Food Quality"],
        })

        report = branch_table(reviews).set_index("branch")

        self.assertNotIn(page_slug, report.index)
        self.assertEqual(report.loc["Main", "reviews"], 2)
        self.assertEqual(report.loc["Koyambedu", "reviews"], 1)
        self.assertEqual(int(report["reviews"].sum()), len(reviews))


if __name__ == "__main__":
    unittest.main()

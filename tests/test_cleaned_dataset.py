"""Tests for loading the already-cleaned review file without further cleaning."""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "scripts_new792026" / "empathy_engine (5)" / "empathy_engine"
sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(ROOT / "scripts_new792026"))

import cleaned_dataset  # noqa: E402

ROWS = [
    ["R1", "Google", "A2B", "T. Nagar", "T Nagar", 'Great, "tasty" food 😀', "2026-01-05", "No"],
    ["R2", "Zomato", "Geetham", "T. Nagar", "T Nagar", 'Great, "tasty" food 😀', "2026-01-05", "No"],
    ["R3", "Google", "Geetham", "Unspecified", "x", "ok", "2026-02-01", "No"],
    ["R4", "Google", "Geetham", "Unspecified", "x", "ok", "2026-02-01", "No"],
]


class CleanedDatasetTests(unittest.TestCase):
    def write_csv(self, folder):
        path = Path(folder) / "All_Reviews_Cleaned.csv"
        pd.DataFrame(ROWS, columns=cleaned_dataset.COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def test_loads_every_row_untouched_as_text(self):
        with tempfile.TemporaryDirectory() as folder:
            df = cleaned_dataset.load_cleaned_reviews(self.write_csv(folder))

        self.assertEqual(len(df), 4)  # duplicates and short reviews are kept
        self.assertEqual(df.loc[0, "Review"], 'Great, "tasty" food 😀')
        self.assertEqual(df.loc[0, "Date"], "2026-01-05")

    def test_branch_count_is_distinct_restaurant_branch_pairs_without_unspecified(self):
        df = pd.DataFrame(ROWS, columns=cleaned_dataset.COLUMNS)

        self.assertEqual(cleaned_dataset.review_count(df), 4)
        self.assertEqual(cleaned_dataset.branch_count(df), 2)
        self.assertEqual(cleaned_dataset.branches_for(df, "Geetham"), ["T. Nagar"])

    def test_warns_when_counts_differ_from_clean_reviews_output(self):
        df = pd.DataFrame(ROWS, columns=cleaned_dataset.COLUMNS)
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / cleaned_dataset.EXPECTED_NAME).write_text('{"rows": 5, "branches": 2}')
            with self.assertLogs("empathy.cleaned_dataset", level="WARNING") as logs:
                warnings = cleaned_dataset.check_counts(df, Path(folder))

        self.assertEqual(len(warnings), 1)
        self.assertIn("loaded 4, clean_reviews.py reported 5", logs.output[0])

    def test_shipped_file_matches_expected_counts(self):
        path = cleaned_dataset.find_cleaned_file(ROOT)
        if path is None:
            self.skipTest("cleaned review file not present")

        df = cleaned_dataset.load_cleaned_reviews(path)

        self.assertEqual(cleaned_dataset.check_counts(df, ROOT), [])
        self.assertEqual((len(df), cleaned_dataset.branch_count(df)), (6146, 103))


if __name__ == "__main__":
    unittest.main()

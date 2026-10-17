"""Tests for the live issue-classification model and staff corrections."""
import sys
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ENGINE_DIR = (Path(__file__).resolve().parents[1] / "scripts_new792026"
              / "empathy_engine (5)" / "empathy_engine")
sys.path.insert(0, str(ENGINE_DIR))

from engine import db  # noqa: E402
from engine import classifier  # noqa: E402


class FixedVectorizer:
    def transform(self, texts):
        return texts


class FixedModel:
    def __init__(self, probabilities):
        self.classes_ = np.array(["Parking & Accessibility", "Billing & Online Ordering Issues"])
        self.probabilities = probabilities

    def predict_proba(self, texts):
        return np.tile(self.probabilities, (len(texts), 1))


def review(text, rating=1, issue=None):
    return pd.DataFrame({
        "text_clean": [text],
        "rating": [rating],
        "issue": [issue],
    })


class LiveIssueClassifierTests(unittest.TestCase):
    def test_high_confidence_prediction_uses_operational_issue_label(self):
        model = FixedModel([0.9, 0.1])
        with patch.object(classifier, "_issue_model", return_value=(model, FixedVectorizer())):
            result = classifier.classify(review("service was very bad"))

        self.assertEqual(result.loc[0, "issue"], "Ambience & Location")
        self.assertEqual(result.loc[0, "issue_source"], "ml_model")
        self.assertAlmostEqual(result.loc[0, "issue_confidence"], 0.9)

    def test_low_confidence_prediction_falls_back_to_rules_and_is_reviewable(self):
        model = FixedModel([0.55, 0.45])
        with patch.object(classifier, "_issue_model", return_value=(model, FixedVectorizer())):
            result = classifier.classify(review("waiting too long for the order"))

        self.assertEqual(result.loc[0, "issue"], "Slow Service & Staff Negligence")
        self.assertEqual(result.loc[0, "issue_source"], "rules_low_confidence")
        self.assertAlmostEqual(result.loc[0, "issue_confidence"], 0.55)

    def test_positive_reviews_and_safety_flags_bypass_model(self):
        with patch.object(classifier, "_issue_model", side_effect=AssertionError("model should be bypassed")):
            positive = classifier.classify(review("excellent food and friendly staff", rating=5))
            safety = classifier.classify(review("food poisoning after the meal", rating=5))

        self.assertEqual(positive.loc[0, "issue"], "Overall Positive Experience")
        self.assertEqual(positive.loc[0, "issue_source"], "rules")
        self.assertEqual(safety.loc[0, "issue"], "Poor Experience / Hygiene Complaints")
        self.assertEqual(safety.loc[0, "issue_source"], "rules")
        self.assertEqual(safety.loc[0, "red_flag"], 1)

    def test_provided_issue_labels_are_preserved(self):
        result = classifier.classify(review("poor service", issue="Human-supplied label"))

        self.assertEqual(result.loc[0, "issue"], "Human-supplied label")
        self.assertEqual(result.loc[0, "issue_source"], "file")
        self.assertTrue(pd.isna(result.loc[0, "issue_confidence"]))

    def test_missing_model_uses_rule_based_classifier(self):
        with patch.object(classifier, "_issue_model", return_value=None):
            result = classifier.classify(review("waiting too long for the order"))

        self.assertEqual(result.loc[0, "issue"], "Slow Service & Staff Negligence")
        self.assertEqual(result.loc[0, "issue_source"], "rules")
        self.assertTrue(pd.isna(result.loc[0, "issue_confidence"]))

    def test_repository_model_is_wired_into_the_live_classifier(self):
        self.assertTrue(classifier.ISSUE_MODEL_PATH.is_file())
        result = classifier.classify(review("The wait was terribly slow and staff ignored us"))

        self.assertIn(result.loc[0, "issue_source"], {"ml_model", "rules_low_confidence"})
        self.assertIn(result.loc[0, "issue"], set(classifier.MODEL_LABEL_MAP.values()))
        self.assertGreaterEqual(result.loc[0, "issue_confidence"], 0)
        self.assertLessEqual(result.loc[0, "issue_confidence"], 1)

    def test_database_migration_persists_confidence_and_staff_corrections(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "empathy.db"
            with patch.object(db, "DB_PATH", db_path):
                legacy_schema = db.SCHEMA.replace(
                    "    issue_confidence REAL,                  -- model confidence estimate; NULL for rules / supplied labels\n",
                    "",
                )
                with sqlite3.connect(db_path) as connection:
                    connection.executescript(legacy_schema)

                db.init_db()
                with db.connect() as connection:
                    columns = {row["name"] for row in connection.execute("PRAGMA table_info(reviews)")}
                self.assertIn("issue_confidence", columns)

                classified = pd.DataFrame([{
                    "review_hash": "sample-hash",
                    "source": "Google",
                    "restaurant": "Geetham",
                    "branch": "Main",
                    "review_date": "2026-10-01",
                    "rating": 1,
                    "text": "Poor service",
                    "text_clean": "poor service",
                    "issue": "Service Quality",
                    "issue_source": "rules_low_confidence",
                    "issue_confidence": 0.42,
                    "sentiment": "negative",
                    "emotion": "neutral",
                    "urgency": 5,
                    "red_flag": 0,
                }])
                self.assertEqual(db.insert_reviews(classified, "test-batch"), 1)
                db.update_issue_label(1, "Slow Service & Staff Negligence", 6)

                with db.connect() as connection:
                    row = connection.execute(
                        "SELECT issue, issue_source, issue_confidence, urgency FROM reviews WHERE id=1"
                    ).fetchone()
                self.assertEqual(row["issue"], "Slow Service & Staff Negligence")
                self.assertEqual(row["issue_source"], "human")
                self.assertIsNone(row["issue_confidence"])
                self.assertEqual(row["urgency"], 6)


if __name__ == "__main__":
    unittest.main()

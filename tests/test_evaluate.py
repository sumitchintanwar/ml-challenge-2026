"""
Unit tests for src/evaluate.py
Tests the official competition macro-averaged F0.5 metric, singleton scoring,
and entity-level evaluation logic.
"""

import unittest
from src.evaluate import compute_entity_f05, evaluate_entity_predictions


class TestEntityEvaluation(unittest.TestCase):

    def test_singleton_correct(self):
        # Empty true set, empty pred set -> perfect singleton score
        f05, prec, rec = compute_entity_f05(set(), set())
        self.assertEqual(f05, 1.0)
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 1.0)

    def test_singleton_false_positive(self):
        # Empty true set, non-empty pred set -> 0.0 score (false merge)
        f05, prec, rec = compute_entity_f05(set(), {"cand_1"})
        self.assertEqual(f05, 0.0)
        self.assertEqual(prec, 0.0)
        self.assertEqual(rec, 1.0)

    def test_matched_empty_prediction(self):
        # True matches exist, but model predicted nothing -> 0.0 score
        f05, prec, rec = compute_entity_f05({"true_1", "true_2"}, set())
        self.assertEqual(f05, 0.0)
        self.assertEqual(prec, 0.0)
        self.assertEqual(rec, 0.0)

    def test_matched_disjoint_prediction(self):
        # Predictions exist but zero overlap with ground truth -> 0.0 score
        f05, prec, rec = compute_entity_f05({"true_1"}, {"cand_other"})
        self.assertEqual(f05, 0.0)
        self.assertEqual(prec, 0.0)
        self.assertEqual(rec, 0.0)

    def test_matched_exact(self):
        # Perfect exact prediction
        f05, prec, rec = compute_entity_f05({"true_1", "true_2"}, {"true_1", "true_2"})
        self.assertEqual(f05, 1.0)
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 1.0)

    def test_matched_precision_vs_recall_weighting(self):
        # F0.5 = (1.25 * P * R) / (0.25 * P + R)
        # Case A: Precision = 1.0, Recall = 0.5 (True: {A, B}, Pred: {A})
        f05_a, prec_a, rec_a = compute_entity_f05({"A", "B"}, {"A"})
        self.assertAlmostEqual(prec_a, 1.0)
        self.assertAlmostEqual(rec_a, 0.5)
        # (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 5/6 = 0.8333
        self.assertAlmostEqual(f05_a, 5.0 / 6.0)

        # Case B: Precision = 0.5, Recall = 1.0 (True: {A}, Pred: {A, B})
        f05_b, prec_b, rec_b = compute_entity_f05({"A"}, {"A", "B"})
        self.assertAlmostEqual(prec_b, 0.5)
        self.assertAlmostEqual(rec_b, 1.0)
        # (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 5/9 = 0.5556
        self.assertAlmostEqual(f05_b, 5.0 / 9.0)

        # F0.5 must penalize low precision more heavily than low recall:
        self.assertGreater(f05_a, f05_b)

    def test_evaluate_entity_predictions_macro(self):
        # 3 entities:
        # e1: Singleton correctly predicted empty -> F0.5 = 1.0
        # e2: Exact match -> F0.5 = 1.0
        # e3: Complete miss -> F0.5 = 0.0
        gt_dict = {
            "e1": set(),
            "e2": {"cand_a"},
            "e3": {"cand_b"},
        }
        preds_dict = {
            "e2": {"cand_a"},
            "e3": {"cand_wrong"},
        }
        eval_ids = ["e1", "e2", "e3"]
        country_map = {"e1": "US", "e2": "US", "e3": "India"}

        metrics = evaluate_entity_predictions(gt_dict, preds_dict, eval_ids, country_map)

        # Macro average = (1.0 + 1.0 + 0.0) / 3 = 0.6667
        self.assertAlmostEqual(metrics["macro_f05"], 0.6667, places=4)
        self.assertEqual(metrics["total_entities"], 3)
        self.assertEqual(metrics["singletons_total"], 1)
        self.assertEqual(metrics["singletons_accuracy"], 1.0)
        self.assertEqual(metrics["matched_total"], 2)
        self.assertAlmostEqual(metrics["matched_macro_f05"], 0.5, places=4)

        # Check country breakdown
        self.assertIn("US", metrics["by_country"])
        self.assertIn("India", metrics["by_country"])
        self.assertEqual(metrics["by_country"]["US"]["entities"], 2)
        self.assertAlmostEqual(metrics["by_country"]["US"]["macro_f05"], 1.0)
        self.assertEqual(metrics["by_country"]["India"]["entities"], 1)
        self.assertAlmostEqual(metrics["by_country"]["India"]["macro_f05"], 0.0)


if __name__ == "__main__":
    unittest.main()

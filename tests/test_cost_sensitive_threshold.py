"""Regression checks for the readable threshold search used by notebook 03b."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from secom_cost_sensitive import CostScenario, select_cost_threshold


class BasicThresholdTests(unittest.TestCase):
    def test_missed_fail_cost_changes_the_selected_action(self):
        labels = [0, 0, 0, 1, 1]
        scores = [0.1, 0.4, 0.8, 0.3, 0.9]
        result = select_cost_threshold(labels, scores, CostScenario("emc_r10", 1, 10))
        self.assertEqual(result["threshold"], 0.3)
        self.assertEqual((result["fp"], result["fn"]), (2, 0))
        self.assertEqual(result["total_cost"], 2)
        self.assertEqual(result["cost_per_product"], 0.4)

    def test_identical_scores_cannot_be_split_and_ties_use_higher_threshold(self):
        result = select_cost_threshold([0, 1], [0.7, 0.7], CostScenario("emc", 1, 10))
        self.assertEqual(result["threshold"], 0.7)
        self.assertEqual((result["fp"], result["fn"]), (1, 0))

    def test_always_pass_must_include_scores_equal_to_one(self):
        result = select_cost_threshold([0, 1], [1.0, 1.0], CostScenario("expensive_fp", 20, 1))
        self.assertGreater(result["threshold"], 1)
        self.assertEqual(result["total_cost"], 1)
        self.assertEqual(result["n_inspections"], 0)

    def test_zero_cost_tie_prefers_fewer_predicted_fail(self):
        result = select_cost_threshold([0, 1], [0.2, 0.8], CostScenario("zero", 0, 0))
        self.assertEqual(result["n_inspections"], 0)
        self.assertEqual(result["threshold"], np.nextafter(1.0, np.inf))

    def test_nonzero_true_positive_cost_is_still_supported(self):
        result = select_cost_threshold([0, 1], [0.1, 0.9], CostScenario("review", 1, 10, tp_cost=20))
        self.assertGreater(result["threshold"], 1)
        self.assertEqual(result["total_cost"], 10)


if __name__ == "__main__":
    unittest.main()

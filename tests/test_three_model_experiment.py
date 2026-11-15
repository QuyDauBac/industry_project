"""Numerical and small CUDA integration checks for the three-model experiment.

Run from the repository root with:
    py -3.11 -X utf8 -m unittest discover -s tests -p test_three_model_experiment.py -v
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import secom_three_model_experiment as experiment


class CostThresholdTests(unittest.TestCase):
    def test_one_hundred_random_curves_match_brute_force(self):
        rng = np.random.default_rng(381)
        for case in range(100):
            n = int(rng.integers(8, 80))
            labels = (rng.random(n) < 0.25).astype(int)
            labels[:2] = [0, 1]
            if case % 2:
                scores = rng.choice([0.0, 0.05, 0.2, 0.5, 0.9, 1.0], size=n)
            else:
                scores = rng.random(n)
                scores[:2] = [0.0, 1.0]
            curve = experiment.cost_threshold_curve(labels, scores)
            thresholds = np.unique(np.r_[0.0, scores, np.nextafter(1.0, np.inf)])
            np.testing.assert_array_equal(curve["threshold"].to_numpy(), thresholds)
            brute = []
            for row in curve.to_dict(orient="records"):
                predicted = scores >= row["threshold"]
                tp = int(np.count_nonzero(predicted & (labels == 1)))
                fp = int(np.count_nonzero(predicted & (labels == 0)))
                fn = int(np.count_nonzero(~predicted & (labels == 1)))
                tn = int(np.count_nonzero(~predicted & (labels == 0)))
                total = tp + fp + 10 * fn
                with self.subTest(case=case, threshold=row["threshold"]):
                    for key, expected in {"tp": tp, "fp": fp, "fn": fn, "tn": tn}.items():
                        self.assertEqual(row[key], expected)
                    self.assertEqual(row["n_inspections"], tp + fp)
                    self.assertEqual(row["total_cost"], total)
                    self.assertAlmostEqual(row["expected_cost"], total / n)
                    self.assertAlmostEqual(row["cost_per_1000"], 1000 * total / n)
                brute.append((total, tp + fp, -row["threshold"]))
            expected_threshold = -min(brute)[2]
            self.assertEqual(experiment.choose_cost_threshold(curve), expected_threshold)
            self.assertEqual(int(curve.iloc[0]["n_inspections"]), n)
            self.assertEqual(int(curve.iloc[-1]["n_inspections"]), 0)
            self.assertGreater(float(curve.iloc[-1]["threshold"]), 1.0)

    def test_cost_ties_prefer_fewer_inspections_then_higher_threshold(self):
        labels = np.r_[1, np.zeros(9, dtype=int)]
        curve = experiment.cost_threshold_curve(labels, np.full(10, 0.5))
        # Reviewing all ten and missing the one Fail both cost ten units.
        self.assertEqual(experiment.choose_cost_threshold(curve), np.nextafter(1.0, np.inf))
        curve = experiment.cost_threshold_curve(np.array([1, 1, 0]), np.full(3, 0.3))
        # Thresholds 0 and .3 produce the same action and cost; choose .3.
        self.assertEqual(experiment.choose_cost_threshold(curve), 0.3)

    def test_full_expected_cost_includes_review_of_true_positives(self):
        labels = np.array([0, 0, 1, 1])
        result = experiment.evaluate_scores(labels, np.array([0.2, 0.8, 0.3, 0.9]), 0.5)
        self.assertEqual((result["tn"], result["fp"], result["fn"], result["tp"]), (1, 1, 1, 1))
        self.assertEqual(result["total_cost"], 12)
        self.assertEqual(result["expected_cost"], 3.0)
        self.assertAlmostEqual(result["emc_error_only"], 11 / 4)
        self.assertAlmostEqual(
            result["expected_cost"] - result["emc_error_only"], result["tp"] / len(labels)
        )

    def test_preprocessor_matches_notebook_02_without_extra_selection(self):
        preprocessing = experiment.make_preprocessor()
        self.assertEqual(list(preprocessing.named_steps), ["missing_rate", "imputer", "constant", "scaler"])
        self.assertEqual(
            [type(step).__name__ for _, step in preprocessing.steps],
            ["MissingRateSelector", "SimpleImputer", "VarianceThreshold", "StandardScaler"],
        )
        self.assertEqual(preprocessing.named_steps["missing_rate"].threshold, 0.70)
        self.assertEqual(preprocessing.named_steps["imputer"].strategy, "median")
        self.assertEqual(preprocessing.named_steps["constant"].threshold, 0.0)


class SmallCudaExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(91)
        indices = np.arange(120)
        labels = (indices % 5 == 0).astype(int)
        values = rng.normal(size=(120, 8))
        values[labels == 1, :2] += 1.4
        values[indices % 7 == 0, 5] = np.nan
        values[:, 6] = np.nan
        values[indices % 10 == 0, 6] = 2.0
        values[:, 7] = 3.0
        X = pd.DataFrame(values, index=indices, columns=[f"f{i}" for i in range(8)])
        y = pd.Series(labels, index=indices)
        cls.X_train, cls.X_test = X.iloc[:90], X.iloc[90:]
        cls.y_train, cls.y_test = y.iloc[:90], y.iloc[90:]
        cls.testing_root = (WORKSPACE / "artifacts" / "testing").resolve()
        cls.testing_root.relative_to(WORKSPACE.resolve())
        cls.testing_root.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="three_model_test_", dir=str(cls.testing_root))
        cls.output_dir = Path(cls.temporary.name).resolve()
        cls.output_dir.relative_to(cls.testing_root)
        cls.addClassCleanup(cls._safe_cleanup)
        cls.candidates = [
            experiment.Candidate("xgb_smoke", "XGBoost", {"max_depth": 2, "min_child_weight": 1}),
            experiment.Candidate("lr_smoke", "LogReg_L1", {"C": 0.1}),
            experiment.Candidate("rf_smoke", "RF_balanced", {
                "n_estimators": 3, "max_depth": 2, "min_samples_leaf": 2,
            }),
        ]
        cls.frozen_digests_at_test = []
        original_evaluate = experiment.evaluate_scores

        def assert_frozen_before_evaluating_test(y_true, scores, threshold):
            if len(y_true) == len(cls.y_test):
                frozen = cls.output_dir / "frozen_policies_before_test.json"
                if not frozen.is_file():
                    raise AssertionError("Test labels evaluated before policy freezing")
                if (cls.output_dir / "results.csv").exists():
                    raise AssertionError("Unexpected earlier holdout results in fresh experiment")
                cls.frozen_digests_at_test.append(hashlib.sha256(frozen.read_bytes()).hexdigest())
            return original_evaluate(y_true, scores, threshold)

        with patch.object(experiment, "evaluate_scores", side_effect=assert_frozen_before_evaluating_test):
            cls.output = experiment.run_experiment(
                cls.X_train, cls.y_train, cls.X_test, cls.y_test,
                seeds=(7,), output_dir=cls.output_dir, verbose=False,
                n_splits=3, candidate_grid=cls.candidates,
                xgb_round_cap=20, early_stopping_patience=3,
            )

    @classmethod
    def _safe_cleanup(cls):
        # Verify the absolute recursive-cleanup target immediately before deletion.
        target = Path(cls.temporary.name).resolve()
        target.relative_to(cls.testing_root)
        target.relative_to(WORKSPACE.resolve())
        if target == cls.testing_root or target == WORKSPACE.resolve():
            raise AssertionError("Refusing to clean a non-temporary parent directory")
        cls.temporary.cleanup()

    def test_oof_coverage_and_holdout_indices(self):
        oof = self.output["oof_predictions"]
        self.assertEqual(len(oof), 90 * 3)
        self.assertTrue(oof.groupby(["seed", "model", "row_index"]).size().eq(1).all())
        for _, frame in oof.groupby(["seed", "model"]):
            self.assertEqual(set(frame["row_index"]), set(self.X_train.index))
            self.assertEqual(set(frame["fold"]), {1, 2, 3})
        predictions = self.output["predictions"]
        self.assertEqual(len(predictions), 30 * 3)
        self.assertTrue(predictions.groupby(["seed", "model", "row_index"]).size().eq(1).all())
        self.assertEqual(set(predictions["row_index"]), set(self.X_test.index))
        self.assertTrue(set(oof["row_index"]).isdisjoint(predictions["row_index"]))

    def test_frozen_policies_precede_test_and_contain_no_test_metrics(self):
        frozen_path = self.output_dir / "frozen_policies_before_test.json"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        self.assertEqual(len(self.frozen_digests_at_test), 5)  # Three models plus two baselines.
        digest = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
        self.assertEqual(set(self.frozen_digests_at_test), {digest})
        for forbidden in ("n_test_fail", "total_runtime_seconds", "test_expected_cost", "test_average_precision"):
            self.assertNotIn(forbidden, frozen["metadata"])
        for policy in frozen["policies"]:
            for metric in set(experiment.METRIC_NAMES) - {"threshold"}:
                self.assertNotIn(metric, policy)
            self.assertIn("oof_average_precision", policy)
            self.assertIn("oof_expected_cost", policy)
        metadata = self.output["metadata"]
        self.assertFalse(metadata["this_is_nested_cv"])
        self.assertTrue(metadata["oof_metrics_are_selection_statistics"])
        self.assertEqual(metadata["cost_matrix"], {"tn": 0, "fp": 1, "tp": 1, "fn": 10})

    def test_cost_arithmetic_and_serialized_prediction_round_trip(self):
        for table_name in ("results", "candidate_search", "baselines"):
            for row in self.output[table_name].to_dict(orient="records"):
                n = row["tp"] + row["fp"] + row["fn"] + row["tn"]
                total = row["tp"] + row["fp"] + 10 * row["fn"]
                self.assertEqual(row["total_cost"], total)
                self.assertAlmostEqual(row["expected_cost"], total / n)
                self.assertAlmostEqual(row["cost_per_1000"], 1000 * total / n)
                self.assertAlmostEqual(row["emc_error_only"], (row["fp"] + 10 * row["fn"]) / n)
        for policy in self.output["policies"].to_dict(orient="records"):
            bundle = joblib.load(self.output_dir / policy["model_path"])
            self.assertEqual(list(bundle["preprocessor"].named_steps), ["missing_rate", "imputer", "constant", "scaler"])
            self.assertEqual(bundle["preprocessor"].transform(self.X_test).shape, (30, 6))
            scores = experiment.predict_bundle(bundle, self.X_test)
            predictions = self.output["predictions"]
            saved = predictions[predictions["model"].eq(policy["model"])].sort_values("row_index")
            np.testing.assert_allclose(scores, saved["score"].to_numpy(), rtol=0, atol=1e-12)
            if policy["model"] == "XGBoost":
                device = json.loads(bundle["estimator"].get_booster().save_config())["learner"]["generic_param"]["device"]
                self.assertTrue(device.startswith("cuda"), device)

    def test_stopping_sets_are_inside_training_and_disjoint_from_validation(self):
        manifest = json.loads((self.output_dir / "cv_split_manifest.json").read_text(encoding="utf-8"))
        folds = {(entry["seed"], entry["fold"]): entry for entry in manifest}
        self.assertEqual(len(folds), 3)
        seen_validation = []
        for entry in manifest:
            fit, valid = set(entry["fit_rows"]), set(entry["valid_rows"])
            self.assertTrue(fit.isdisjoint(valid))
            self.assertEqual(fit | valid, set(self.X_train.index))
            self.assertTrue((fit | valid).isdisjoint(self.X_test.index))
            seen_validation.extend(valid)
        self.assertEqual(sorted(seen_validation), sorted(self.X_train.index))
        for row in self.output["cv_fold_log"].query("model == 'XGBoost'").to_dict(orient="records"):
            cv = folds[(row["seed"], row["fold"])]
            fit, valid = set(cv["fit_rows"]), set(cv["valid_rows"])
            stop_fit, stop_valid = set(row["stop_fit_rows"]), set(row["stop_valid_rows"])
            self.assertTrue(stop_fit.isdisjoint(stop_valid))
            self.assertEqual(stop_fit | stop_valid, fit)
            self.assertTrue((stop_fit | stop_valid).isdisjoint(valid))
            self.assertTrue((stop_fit | stop_valid).isdisjoint(self.X_test.index))
            self.assertEqual(len(stop_fit), row["n_stop_fit"])
            self.assertEqual(len(stop_valid), row["n_stop_valid"])
            self.assertTrue(row["gpu_device"].startswith("cuda"))
            self.assertGreaterEqual(row["best_round"], 1)
            self.assertLessEqual(row["best_round"], 20)

    def test_changed_feature_order_is_rejected_before_work(self):
        with self.assertRaisesRegex(ValueError, "feature order"):
            experiment.run_experiment(
                self.X_train, self.y_train, self.X_test.iloc[:, ::-1], self.y_test,
                seeds=(7,), output_dir=self.output_dir / "must_not_exist",
                n_splits=3, candidate_grid=self.candidates,
                xgb_round_cap=20, early_stopping_patience=3,
            )
        self.assertFalse((self.output_dir / "must_not_exist").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)

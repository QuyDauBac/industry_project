"""Five-seed XGB/L1-logistic/weighted-RF experiment with a fixed holdout.

The four preprocessing steps are exactly those in notebook 02. Candidate
selection uses training OOF Average Precision; threshold selection minimizes
TP + FP + 10*FN on those OOF scores. These are development-selection statistics,
not nested/unbiased estimates. All policies are frozen before test evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc, average_precision_score, matthews_corrcoef, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from secom_preprocessing import MissingRateSelector
from secom_utils import MISSING_THRESHOLD, evaluate_classifier


MODEL_NAMES = ("XGBoost", "LogReg_L1", "RF_balanced")
DEFAULT_SEEDS = (13, 42, 123, 2024, 2026)
METRIC_NAMES = (
    "average_precision", "pr_auc_trapezoid", "roc_auc", "accuracy", "precision",
    "recall", "f1", "balanced_accuracy", "ber", "fpr", "fnr", "specificity",
    "g_mean", "mcc", "alarm_rate", "n_inspections", "tp", "fp", "fn", "tn",
    "total_cost", "expected_cost", "cost_per_1000", "emc_error_only", "threshold",
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    model: str
    params: dict


def make_candidates(rf_trees: int = 300) -> list[Candidate]:
    candidates = []
    for depth in (2, 3):
        for child_weight in (1, 5):
            candidates.append(Candidate(
                f"xgb_d{depth}_w{child_weight}", "XGBoost",
                {"max_depth": depth, "min_child_weight": child_weight},
            ))
    for strength in (0.01, 0.1, 1.0):
        candidates.append(Candidate(
            f"lr_l1_c{strength:g}".replace(".", "p"), "LogReg_L1", {"C": strength},
        ))
    for depth in (5, None):
        for leaf in (1, 5):
            candidates.append(Candidate(
                f"rf_d{depth}_leaf{leaf}", "RF_balanced",
                {"max_depth": depth, "min_samples_leaf": leaf,
                 "n_estimators": int(rf_trees)},
            ))
    return candidates


def make_preprocessor() -> Pipeline:
    return Pipeline([
        ("missing_rate", MissingRateSelector(threshold=MISSING_THRESHOLD)),
        ("imputer", SimpleImputer(strategy="median")),
        ("constant", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
    ])


def _validate_xy(X, y):
    labels = np.asarray(y)
    if not isinstance(X, pd.DataFrame) or labels.ndim != 1 or len(X) != len(labels):
        raise ValueError("X must be a DataFrame aligned with one-dimensional y")
    if set(np.unique(labels)) != {0, 1} or not X.index.is_unique:
        raise ValueError("Both binary classes and unique row indices are required")
    if isinstance(y, pd.Series) and not X.index.equals(y.index):
        raise ValueError("X/y indices must match in order")
    return labels.astype(int)


def evaluate_scores(y, scores, threshold):
    """Full approved cost matrix: TN=0, FP=1, TP=1, FN=10."""
    labels, scores = np.asarray(y).astype(int), np.asarray(scores, dtype=float)
    metrics = evaluate_classifier(labels, scores, threshold)
    precision, recall, _ = precision_recall_curve(labels, scores)
    metrics["pr_auc_trapezoid"] = float(auc(recall, precision))
    metrics["specificity"] = 1.0 - metrics["fpr"]
    metrics["g_mean"] = float(np.sqrt(metrics["recall"] * metrics["specificity"]))
    metrics["mcc"] = float(matthews_corrcoef(labels, scores >= threshold))
    metrics["n_inspections"] = int(metrics["tp"] + metrics["fp"])
    metrics["total_cost"] = float(metrics["tp"] + metrics["fp"] + 10 * metrics["fn"])
    metrics["expected_cost"] = metrics["total_cost"] / len(labels)
    metrics["cost_per_1000"] = 1000 * metrics["expected_cost"]
    metrics["emc_error_only"] = (metrics["fp"] + 10 * metrics["fn"]) / len(labels)
    return metrics


def cost_threshold_curve(y, scores):
    """Scan PR-curve thresholds plus constant actions, with >= semantics."""
    labels, values = np.asarray(y).astype(int), np.asarray(scores, dtype=float)
    if values.shape != labels.shape or not np.isfinite(values).all():
        raise ValueError("Scores must be finite and aligned")
    if (values < 0).any() or (values > 1).any():
        raise ValueError("Expected classifier scores in [0,1]")
    _, _, pr_thresholds = precision_recall_curve(labels, values)
    thresholds = np.unique(np.r_[0.0, pr_thresholds, np.nextafter(1.0, np.inf)])
    order = np.argsort(values, kind="stable")
    below = np.searchsorted(values[order], thresholds, side="left")
    fn = np.r_[0, np.cumsum(labels[order])][below]
    tn = below - fn
    tp = int(labels.sum()) - fn
    fp = len(labels) - int(labels.sum()) - tn
    inspections = tp + fp
    precision = np.divide(tp, inspections, out=np.zeros(len(tp), dtype=float), where=inspections > 0)
    recall = tp / int(labels.sum())
    f1_denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, f1_denominator, out=np.zeros(len(tp), dtype=float), where=f1_denominator > 0)
    ba = (recall + tn / int((labels == 0).sum())) / 2
    cost = inspections + 10 * fn
    return pd.DataFrame({
        "threshold": thresholds, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_inspections": inspections, "total_cost": cost,
        "expected_cost": cost / len(labels), "cost_per_1000": cost / len(labels) * 1000,
        "precision": precision, "recall": recall, "f1": f1,
        "balanced_accuracy": ba, "ber": 1 - ba,
    })


def choose_cost_threshold(curve):
    chosen = curve.sort_values(
        ["total_cost", "n_inspections", "threshold"], ascending=[True, True, False],
        kind="stable",
    ).iloc[0]
    return float(chosen["threshold"])


def _xgb_classifier(candidate, seed, labels, rounds, stopping_rounds=None):
    params = dict(
        **candidate.params, objective="binary:logistic", tree_method="hist", device="cuda:0",
        n_estimators=int(rounds), learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=5.0, reg_alpha=0.1, max_bin=128, n_jobs=4, random_state=int(seed),
        scale_pos_weight=float((labels == 0).sum() / (labels == 1).sum()),
    )
    if stopping_rounds is not None:
        params.update({
            "eval_metric": average_precision_score,
            "callbacks": [xgb.callback.EarlyStopping(
                rounds=stopping_rounds, metric_name="average_precision_score",
                data_name="validation_1", maximize=True, save_best=True,
            )],
        })
    return xgb.XGBClassifier(**params)


def predict_bundle(bundle, X):
    transformed = bundle["preprocessor"].transform(X)
    model = bundle["estimator"]
    if bundle["model"] == "XGBoost":
        # DMatrix performs explicit CPU->GPU transfer, avoiding inplace_predict
        # device-mismatch fallback with a NumPy input and a CUDA booster.
        return model.get_booster().predict(xgb.DMatrix(transformed, nthread=4))
    column = int(np.flatnonzero(model.classes_ == 1)[0])
    return model.predict_proba(transformed)[:, column]


def fit_bundle(candidate, X, labels, seed, rounds=0):
    start = time.perf_counter()
    preprocessor = make_preprocessor()
    transformed = preprocessor.fit_transform(X)
    if candidate.model == "XGBoost":
        estimator = _xgb_classifier(candidate, seed, labels, rounds)
    elif candidate.model == "LogReg_L1":
        estimator = LogisticRegression(
            **candidate.params, solver="liblinear", l1_ratio=1.0, class_weight="balanced",
            max_iter=5000, tol=1e-3, random_state=int(seed),
        )
    else:
        estimator = RandomForestClassifier(
            **candidate.params, class_weight="balanced", max_features="sqrt",
            n_jobs=4, random_state=int(seed),
        )
    with warnings.catch_warnings(record=True) as recorded, threadpool_limits(limits=4):
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(transformed, labels)
    convergence_warning = "; ".join(str(w.message) for w in recorded if issubclass(w.category, ConvergenceWarning))
    other_warnings = "; ".join(str(w.message) for w in recorded if not issubclass(w.category, ConvergenceWarning))
    device = "cpu"
    if candidate.model == "XGBoost":
        config = json.loads(estimator.get_booster().save_config())
        device = config["learner"]["generic_param"]["device"]
        if not device.startswith("cuda"):
            raise RuntimeError(f"XGBoost requested GPU but actually used {device}")
    bundle = {"preprocessor": preprocessor, "estimator": estimator,
              "model": candidate.model, "candidate_id": candidate.candidate_id}
    health = {
        "fit_seconds": time.perf_counter() - start,
        "n_features": int(transformed.shape[1]),
        "iterations": int(estimator.n_iter_.max()) if candidate.model == "LogReg_L1" else 0,
        "nonzero_features": int(np.count_nonzero(estimator.coef_)) if candidate.model == "LogReg_L1" else 0,
        "boosting_rounds": int(rounds) if candidate.model == "XGBoost" else 0,
        "convergence_warning": convergence_warning, "other_warnings": other_warnings,
        "gpu_device": device,
    }
    return bundle, health


def _early_stopping_rounds(candidate, X, labels, seed, cap, patience):
    """Stopping labels belong ONLY to the current CV training partition."""
    fit, stop = train_test_split(
        np.arange(len(labels)), test_size=0.20, stratify=labels, random_state=int(seed),
    )
    preprocessor = make_preprocessor()
    pilot_X = preprocessor.fit_transform(X.iloc[fit])
    stop_X = preprocessor.transform(X.iloc[stop])
    estimator = _xgb_classifier(candidate, seed, labels[fit], cap, patience)
    with threadpool_limits(limits=4):
        estimator.fit(pilot_X, labels[fit], eval_set=[(pilot_X, labels[fit]), (stop_X, labels[stop])], verbose=False)
    pilot_device = json.loads(estimator.get_booster().save_config())["learner"]["generic_param"]["device"]
    if not pilot_device.startswith("cuda"):
        raise RuntimeError(f"XGB early-stopping pilot requested GPU but used {pilot_device}")
    history = estimator.evals_result()
    train_ap = history["validation_0"]["average_precision_score"]
    stop_ap = history["validation_1"]["average_precision_score"]
    best_round = int(estimator.best_iteration) + 1
    curves = [
        {"round": i + 1, "train_ap": float(train_value), "stop_ap": float(stop_value), "best_round": best_round}
        for i, (train_value, stop_value) in enumerate(zip(train_ap, stop_ap))
    ]
    return best_round, curves, {
        "n_stop_fit": len(fit), "n_stop_valid": len(stop),
        "pilot_gpu_device": pilot_device,
        "hit_round_cap": len(stop_ap) >= cap,
        "stop_fit_rows": X.index[fit].tolist(), "stop_valid_rows": X.index[stop].tolist(),
    }


def _summary(frame):
    rows = []
    for model, group in frame.groupby("model", sort=False):
        row = {"model": model, "n_runs": int(group["seed"].nunique())}
        for metric in METRIC_NAMES:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _gpu_environment():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else "nvidia-smi unavailable"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "nvidia-smi unavailable"


def run_experiment(
    X_train, y_train, X_test, y_test, seeds=DEFAULT_SEEDS, output_dir="artifacts/three_model_five_seeds",
    split_seed=42, verbose=True, *, n_splits=5, candidate_grid=None,
    xgb_round_cap=1000, early_stopping_patience=50,
):
    train_labels = _validate_xy(X_train, y_train)
    if not X_train.index.intersection(X_test.index).empty:
        raise ValueError("Train and test row indices must be disjoint")
    if not X_train.columns.equals(X_test.columns):
        raise ValueError("Train/test feature order differs")
    if len(set(seeds)) != len(seeds) or len(seeds) == 0:
        raise ValueError("Seeds must be nonempty and unique")
    if np.bincount(train_labels).min() < n_splits:
        raise ValueError("Not enough minority observations for CV")
    candidates = list(candidate_grid) if candidate_grid is not None else make_candidates()
    if {c.model for c in candidates} != set(MODEL_NAMES):
        raise ValueError("Candidate grid must contain all three model families")
    if len({c.candidate_id for c in candidates}) != len(candidates):
        raise ValueError("Candidate IDs must be unique")
    destination = Path(output_dir)
    (destination / "models").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    search_rows, fold_rows, history_rows, selected_curves = [], [], [], []
    oof_frames, development_rows, policies = [], [], []
    split_records = []
    metadata = {
        "split_seed": int(split_seed), "experiment_seeds": list(map(int, seeds)),
        "n_cv_folds": n_splits, "same_holdout_for_all_runs": True,
        "test_runs_are_independent_datasets": False, "costs_are_assumptions": True,
        "cost_matrix": {"tn": 0, "fp": 1, "tp": 1, "fn": 10},
        "hyperparameter_selection": "Highest pooled training-OOF sklearn Average Precision; tie lower OOF cost, then candidate ID",
        "threshold_selection": "Minimum OOF TP+FP+10*FN; tie fewer inspections, higher threshold",
        "primary_metric": "expected_cost", "score_is_calibrated_probability": False,
        "oof_metrics_are_selection_statistics": True, "this_is_nested_cv": False,
        "xgb_round_cap": int(xgb_round_cap), "early_stopping_patience": int(early_stopping_patience),
        "xgb_early_stop_metric": "sklearn average_precision_score, maximize=True",
        "xgb_final_rounds": "Median of selected candidate CV best rounds, round half up, minimum 1",
        "preprocessing": ["MissingRateSelector(0.70)", "SimpleImputer(median)", "VarianceThreshold(0)", "StandardScaler"],
        "feature_selection": "None; L1 sparsity is intrinsic to the logistic classifier",
        "kaggle_reference_url": "https://www.kaggle.com/code/thienphuc2201/industrial-ai-test",
        "kaggle_reference_status": "Not retrieved: broad Kaggle access failure; methods not verified or claimed reproduced",
        "method_sources": ["02_preprocessing.ipynb", "03_modeling.ipynb", "04_evaluation.ipynb", "official sklearn/XGBoost documentation"],
        "gpu_environment": _gpu_environment(), "xgboost_build_info": xgb.build_info(),
        "python_version": platform.python_version(),
        "package_versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "matplotlib", "joblib")},
        "n_train": len(X_train), "n_train_fail": int(train_labels.sum()), "n_test": len(X_test),
        "train_row_indices": list(map(int, X_train.index)), "test_row_indices": list(map(int, X_test.index)),
        "training_data_hash": hashlib.sha256(pd.util.hash_pandas_object(X_train, index=True).values.tobytes() + train_labels.tobytes()).hexdigest(),
        "candidate_grid": [{"candidate_id": c.candidate_id, "model": c.model, "params": c.params} for c in candidates],
    }
    # Phase 1: no test labels or test predictions are used in any selection.
    for seed in seeds:
        seed = int(seed)
        folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X_train, train_labels))
        for fold, (fit, valid) in enumerate(folds, start=1):
            split_records.append({"seed": seed, "fold": fold,
                                  "fit_rows": X_train.index[fit].tolist(), "valid_rows": X_train.index[valid].tolist()})
        for model_name in MODEL_NAMES:
            if verbose:
                print(f"Seed {seed} | {model_name}: CV selection", flush=True)
            candidate_outputs = {}
            family_rows = []
            for candidate in (c for c in candidates if c.model == model_name):
                scores = np.full(len(X_train), np.nan)
                fold_ids = np.zeros(len(X_train), dtype=int)
                best_rounds, cv_seconds = [], 0.0
                for fold, (fit, valid) in enumerate(folds, start=1):
                    start_fold = time.perf_counter()
                    X_fit, labels_fit = X_train.iloc[fit], train_labels[fit]
                    details = {"n_stop_fit": 0, "n_stop_valid": 0, "hit_round_cap": False,
                               "stop_fit_rows": [], "stop_valid_rows": []}
                    rounds = 0
                    if model_name == "XGBoost":
                        rounds, curves, details = _early_stopping_rounds(
                            candidate, X_fit, labels_fit, seed + 1000 * fold,
                            xgb_round_cap, early_stopping_patience,
                        )
                        for item in curves:
                            history_rows.append({"seed": seed, "candidate_id": candidate.candidate_id, "fold": fold, **item})
                        best_rounds.append(rounds)
                    bundle, health = fit_bundle(candidate, X_fit, labels_fit, seed, rounds)
                    scores[valid] = predict_bundle(bundle, X_train.iloc[valid])
                    train_scores = predict_bundle(bundle, X_fit)
                    fold_ids[valid] = fold
                    elapsed = time.perf_counter() - start_fold
                    cv_seconds += elapsed
                    fold_rows.append({
                        "seed": seed, "model": model_name, "candidate_id": candidate.candidate_id, "fold": fold,
                        **health, "fit_seconds": elapsed, "best_round": rounds,
                        "n_fit": len(fit), "n_valid": len(valid),
                        "train_ap": float(average_precision_score(labels_fit, train_scores)),
                        "valid_ap": float(average_precision_score(train_labels[valid], scores[valid])),
                        **details,
                    })
                if not np.isfinite(scores).all() or (fold_ids == 0).any():
                    raise RuntimeError("Incomplete OOF predictions")
                curve = cost_threshold_curve(train_labels, scores)
                threshold = choose_cost_threshold(curve)
                metrics = evaluate_scores(train_labels, scores, threshold)
                median_rounds = max(1, int(np.floor(np.median(best_rounds) + 0.5))) if best_rounds else 0
                row = {"seed": seed, "model": model_name, "candidate_id": candidate.candidate_id,
                       "params_json": json.dumps(candidate.params, sort_keys=True), **metrics,
                       "cv_fit_seconds": cv_seconds, "median_boosting_rounds": median_rounds, "selected": False}
                family_rows.append(row)
                candidate_outputs[candidate.candidate_id] = (candidate, scores, fold_ids, curve, median_rounds)
                if verbose:
                    print(f"  {candidate.candidate_id}: OOF AP={metrics['average_precision']:.4f}, cost={metrics['expected_cost']:.4f}", flush=True)
            ranked = pd.DataFrame(family_rows).sort_values(
                ["average_precision", "expected_cost", "candidate_id"], ascending=[False, True, True], kind="stable"
            )
            chosen_id = str(ranked.iloc[0]["candidate_id"])
            for row in family_rows:
                row["selected"] = row["candidate_id"] == chosen_id
                search_rows.append(row)
            selected, scores, fold_ids, curve, rounds = candidate_outputs[chosen_id]
            threshold = choose_cost_threshold(curve)
            oof_metrics = evaluate_scores(train_labels, scores, threshold)
            development_rows.append({"seed": seed, "model": model_name, **oof_metrics})
            oof_frames.append(pd.DataFrame({
                "seed": seed, "model": model_name, "candidate_id": chosen_id, "row_index": X_train.index,
                "fold": fold_ids, "y_true": train_labels, "score": scores,
                "y_pred": (scores >= threshold).astype(int), "threshold": threshold,
            }))
            curve = curve.assign(seed=seed, model=model_name, candidate_id=chosen_id)
            selected_curves.append(curve)
            bundle, health = fit_bundle(selected, X_train, train_labels, seed, rounds)
            model_path = f"models/{model_name}_seed{seed}.joblib"
            joblib.dump(bundle, destination / model_path)
            policy = {
                "seed": seed, "model": model_name, "candidate_id": chosen_id,
                "params_json": json.dumps(selected.params, sort_keys=True), "threshold": threshold,
                "oof_average_precision": oof_metrics["average_precision"],
                "oof_expected_cost": oof_metrics["expected_cost"], **health, "model_path": model_path,
            }
            policies.append(policy)
            if verbose:
                print(f"  Frozen: {chosen_id}, threshold={threshold:.5f}, rounds={rounds}, device={health['gpu_device']}", flush=True)
        # Checkpoint development tables; no holdout results exist at this point.
        pd.DataFrame(search_rows).to_csv(destination / "candidate_search.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(fold_rows).to_csv(destination / "cv_fold_log.csv", index=False, encoding="utf-8-sig")
    development_summary = _summary(pd.DataFrame(development_rows))
    development_winner = development_summary.sort_values(
        ["expected_cost_mean", "average_precision_mean", "model"], ascending=[True, False, True]
    ).iloc[0]["model"]
    metadata["development_winner"] = str(development_winner)
    metadata["development_selection_seconds"] = time.perf_counter() - started
    (destination / "frozen_policies_before_test.json").write_text(
        json.dumps({"metadata": metadata, "policies": policies}, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (destination / "cv_split_manifest.json").write_text(json.dumps(split_records, indent=2), encoding="utf-8")
    # Phase 2: all candidate choices, thresholds and the development winner are frozen.
    test_labels = _validate_xy(X_test, y_test)
    result_rows, prediction_frames = [], []
    for policy in policies:
        bundle = joblib.load(destination / policy["model_path"])
        scores = predict_bundle(bundle, X_test)
        metrics = evaluate_scores(test_labels, scores, policy["threshold"])
        result_rows.append({**policy, **metrics})
        prediction_frames.append(pd.DataFrame({
            "seed": policy["seed"], "model": policy["model"], "row_index": X_test.index,
            "y_true": test_labels, "score": scores, "y_pred": (scores >= policy["threshold"]).astype(int),
            "threshold": policy["threshold"],
        }))
    results = pd.DataFrame(result_rows)
    baselines = pd.DataFrame([
        {"policy": "always_pass", **evaluate_scores(test_labels, np.zeros(len(test_labels)), np.nextafter(1.0, np.inf))},
        {"policy": "review_all", **evaluate_scores(test_labels, np.ones(len(test_labels)), 0.0)},
    ])
    metadata.update({"n_test_fail": int(test_labels.sum()), "total_runtime_seconds": time.perf_counter() - started})
    output = {
        "results": results, "predictions": pd.concat(prediction_frames, ignore_index=True),
        "oof_predictions": pd.concat(oof_frames, ignore_index=True),
        "candidate_search": pd.DataFrame(search_rows), "cv_fold_log": pd.DataFrame(fold_rows),
        "learning_history": pd.DataFrame(history_rows), "threshold_curves": pd.concat(selected_curves, ignore_index=True),
        "summary": _summary(results), "development_summary": development_summary,
        "baselines": baselines, "policies": pd.DataFrame(policies),
    }
    for name, frame in output.items():
        frame.to_csv(destination / f"{name}.csv", index=False, encoding="utf-8-sig")
    (destination / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    output["metadata"] = metadata
    if verbose:
        print(f"Completed {len(results)} fixed-holdout evaluations in {metadata['total_runtime_seconds']:.1f}s.", flush=True)
    return output

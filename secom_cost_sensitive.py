"""Cost-sensitive SECOM experiments, isolated from the original model artifacts.

Costs are relative, explicitly assumed units. No company currency or costs are
inferred. Nested CV evaluates the complete parameter/threshold selection process
on the existing training partition; it never accesses the existing holdout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from secom_preprocessing import MissingRateSelector
from secom_utils import MISSING_THRESHOLD, evaluate_classifier


@dataclass(frozen=True)
class CostScenario:
    name: str
    fp_cost: float
    fn_cost: float
    tp_cost: float = 0.0
    tn_cost: float = 0.0
    family: str = "misclassification"

    def __post_init__(self) -> None:
        costs = np.asarray(
            [self.fp_cost, self.fn_cost, self.tp_cost, self.tn_cost], dtype=float
        )
        if not self.name or not np.isfinite(costs).all() or (costs < 0).any():
            raise ValueError("Scenario names must be nonempty; costs finite and nonnegative")


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    family: str
    params: dict[str, Any]


def make_candidate_specs(
    k_values: Sequence[int] = (20, 40, 80),
    lr_c_values: Sequence[float] = (0.1, 1.0, 10.0),
    rf_depth_values: Sequence[int | None] = (5, None),
    rf_leaf_values: Sequence[int] = (1, 5),
    n_estimators: int = 200,
    rf_n_jobs: int = 1,
) -> list[CandidateSpec]:
    """A small, predeclared search; both classifiers retain balanced weights."""
    candidates = []
    for k in k_values:
        for c_value in lr_c_values:
            candidates.append(CandidateSpec(
                f"lr_k{k}_c{c_value:g}".replace(".", "p"),
                "LR",
                {"k": int(k), "C": float(c_value)},
            ))
        for depth in rf_depth_values:
            for leaf in rf_leaf_values:
                candidates.append(CandidateSpec(
                    f"rf_k{k}_d{depth}_leaf{leaf}",
                    "RF",
                    {"k": int(k), "max_depth": depth,
                     "min_samples_leaf": int(leaf), "n_estimators": int(n_estimators),
                     "n_jobs": int(rf_n_jobs)},
                ))
    if not candidates or len({c.candidate_id for c in candidates}) != len(candidates):
        raise ValueError("Candidate grid must be nonempty with unique configurations")
    return candidates


def build_pipeline(candidate: CandidateSpec, seed: int = 42) -> Pipeline:
    """Every learned preprocessing step is fitted inside the relevant CV fold."""
    params = dict(candidate.params)
    k = params.pop("k")
    if candidate.family == "LR":
        classifier = LogisticRegression(
            **params, max_iter=2000, class_weight="balanced", random_state=seed
        )
    elif candidate.family == "RF":
        classifier = RandomForestClassifier(
            **params, class_weight="balanced", random_state=seed
        )
    else:
        raise ValueError(f"Unsupported family: {candidate.family}")
    return Pipeline([
        ("missing_rate", MissingRateSelector(threshold=MISSING_THRESHOLD)),
        ("imputer", SimpleImputer(strategy="median")),
        ("constant", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=k)),
        ("model", classifier),
    ])


def _validate_data(X: pd.DataFrame, y: Any, n_splits: int) -> np.ndarray:
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != len(X) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Aligned data with both binary labels 0/1 are required")
    if n_splits < 2 or np.bincount(labels.astype(int)).min() < n_splits:
        raise ValueError("Each class must have at least n_splits observations")
    if isinstance(y, pd.Series) and not X.index.equals(y.index):
        raise ValueError("X and y index/order differ")
    if not X.index.is_unique:
        raise ValueError("X must have unique indices for out-of-fold auditing")
    return labels.astype(int)


def _validate_search(candidates: Sequence[CandidateSpec], scenarios: Sequence[CostScenario]) -> None:
    if not candidates or {c.family for c in candidates} != {"LR", "RF"}:
        raise ValueError("Provide at least one LR and one RF candidate")
    if len({c.candidate_id for c in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate IDs")
    if not scenarios or len({s.name for s in scenarios}) != len(scenarios):
        raise ValueError("Scenarios must be nonempty with unique names")


def _total_cost(counts: dict[str, Any], scenario: CostScenario) -> float:
    return float(
        scenario.tp_cost * counts["tp"] + scenario.fp_cost * counts["fp"]
        + scenario.fn_cost * counts["fn"] + scenario.tn_cost * counts["tn"]
    )


def _metrics(y: np.ndarray, scores: np.ndarray, threshold: float, scenario: CostScenario) -> dict:
    metrics = evaluate_classifier(y, scores, threshold)
    n = len(y)
    total = _total_cost(metrics, scenario)
    inspections = int(metrics["tp"] + metrics["fp"])
    metrics.update({
        "n_samples": n, "n_inspections": inspections, "inspection_rate": inspections / n,
        "total_cost": total, "cost_per_product": total / n, "cost_per_1000": total / n * 1000,
    })
    return metrics


def select_cost_threshold(y: Any, scores: Any, scenario: CostScenario) -> dict[str, Any]:
    """Minimize empirical cost on selection data, including the always-Pass action.

    Predictions use score >= threshold. Ties prefer fewer inspections, then a
    higher threshold. Scores must lie in [0, 1] but need not be calibrated.
    Returned metrics are selection statistics, not an unbiased evaluation.
    """
    labels = np.asarray(y)
    values = np.asarray(scores, dtype=float)
    if (labels.ndim != 1 or values.shape != labels.shape or not len(labels)
            or not set(np.unique(labels)).issubset({0, 1}) or not np.isfinite(values).all()
            or (values < 0).any() or (values > 1).any()):
        raise ValueError("Aligned scores in [0, 1] and binary labels are required")
    labels = labels.astype(int)
    order = np.argsort(values, kind="stable")
    sorted_scores, sorted_y = values[order], labels[order]
    # A constant always-Pass policy must remain constant on refitted models too,
    # even if their scores exceed the maximum observed during inner selection.
    thresholds = np.unique(np.r_[0.0, sorted_scores, np.nextafter(1.0, np.inf)])
    below = np.searchsorted(sorted_scores, thresholds, side="left")
    positives_below = np.r_[0, np.cumsum(sorted_y)][below]
    fn = positives_below
    tn = below - fn
    tp = int(labels.sum()) - fn
    fp = len(labels) - int(labels.sum()) - tn
    costs = (scenario.tp_cost * tp + scenario.fp_cost * fp
             + scenario.fn_cost * fn + scenario.tn_cost * tn)
    inspections = tp + fp
    # lexsort uses the last key as primary: cost, inspections, higher threshold.
    best = int(np.lexsort((-thresholds, inspections, costs))[0])
    return _metrics(labels, values, float(thresholds[best]), scenario)


def _selection_search(
    X: pd.DataFrame, y: np.ndarray, candidates: Sequence[CandidateSpec],
    scenarios: Sequence[CostScenario], n_splits: int, seed: int, verbose: bool,
) -> pd.DataFrame:
    _validate_data(X, y, n_splits)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for position, candidate in enumerate(candidates, start=1):
        with threadpool_limits(limits=1):
            # Every fold contains both classes, ordered [0, 1]. The 1 column is Fail.
            scores = cross_val_predict(
                build_pipeline(candidate, seed), X, y, cv=folds,
                method="predict_proba", n_jobs=1,
            )[:, 1]
        for scenario in scenarios:
            row = select_cost_threshold(y, scores, scenario)
            row.update({
                "scenario": scenario.name, "cost_family": scenario.family,
                "candidate_id": candidate.candidate_id, "candidate_family": candidate.family,
                "params_json": json.dumps(candidate.params, sort_keys=True),
                "estimate_type": "selection_only_not_unbiased",
            })
            rows.append(row)
        if verbose and (position == 1 or position % 5 == 0 or position == len(candidates)):
            print(f"  OOF candidates {position}/{len(candidates)}", flush=True)
    return pd.DataFrame(rows)


def _selected_rows(search: pd.DataFrame, scenarios: Sequence[CostScenario]):
    for scenario in scenarios:
        for scope in ("LR", "RF", "cost_selected"):
            eligible = search[search["scenario"].eq(scenario.name)]
            if scope != "cost_selected":
                eligible = eligible[eligible["candidate_family"].eq(scope)]
            ranked = eligible.sort_values(
                ["total_cost", "n_inspections", "threshold", "candidate_id"],
                ascending=[True, True, False, True], kind="stable",
            )
            yield scenario, scope, ranked.iloc[0].to_dict()


def _summarize(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["scenario", "cost_family", "selection_scope"]
    for key, group in fold_metrics.groupby(keys, sort=False):
        counts = {name: int(group[name].sum()) for name in ("tp", "fp", "fn", "tn")}
        tp, fp, fn, tn = (counts[name] for name in ("tp", "fp", "fn", "tn"))
        n, inspected = tp + fp + fn + tn, tp + fp
        precision = tp / inspected if inspected else 0.0
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        total = float(group["total_cost"].sum())
        row = dict(zip(keys, key))
        row.update(counts)
        row.update({
            "n_samples": n, "n_folds": int(group["fold"].nunique()),
            "n_inspections": inspected, "inspection_rate": inspected / n,
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "balanced_accuracy": (recall + specificity) / 2,
            "ber": 1 - (recall + specificity) / 2,
            "total_cost": total, "cost_per_product": total / n,
            "cost_per_1000": total / n * 1000,
            "cost_per_product_fold_std": float(group["cost_per_product"].std(ddof=1)),
            "estimate_type": "nested_outer_evaluation",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def run_nested_cost_cv(
    X: pd.DataFrame, y: Any, candidates: Sequence[CandidateSpec],
    scenarios: Sequence[CostScenario], outer_splits: int = 5,
    inner_splits: int = 3, seed: int = 42, verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Evaluate independent LR/RF searches AND the complete model-selection rule.

    Inner folds select parameters and threshold. Each untouched outer fold is
    scored exactly once per scenario/scope. Models selected for multiple cost
    scenarios are fitted once per outer fold and reused, without label leakage.
    """
    labels = _validate_data(X, y, outer_splits)
    _validate_search(candidates, scenarios)
    by_id = {c.candidate_id: c for c in candidates}
    cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    metric_rows, prediction_frames, searches = [], [], []
    for fold, (fit_positions, valid_positions) in enumerate(cv.split(X, labels), start=1):
        X_fit, X_valid = X.iloc[fit_positions], X.iloc[valid_positions]
        y_fit, y_valid = labels[fit_positions], labels[valid_positions]
        if verbose:
            print(f"Outer fold {fold}/{outer_splits}: fit={len(y_fit)}, validate={len(y_valid)}", flush=True)
        fold_seed = seed + fold
        search = _selection_search(
            X_fit, y_fit, candidates, scenarios, inner_splits, fold_seed, verbose
        )
        search["outer_fold"] = fold
        searches.append(search)
        score_cache = {}
        for scenario, scope, selected in _selected_rows(search, scenarios):
            candidate_id = selected["candidate_id"]
            if candidate_id not in score_cache:
                model = build_pipeline(by_id[candidate_id], fold_seed)
                with threadpool_limits(limits=1):
                    model.fit(X_fit, y_fit)
                    fail_column = int(np.flatnonzero(model.classes_ == 1)[0])
                    score_cache[candidate_id] = model.predict_proba(X_valid)[:, fail_column]
            threshold, scores = float(selected["threshold"]), score_cache[candidate_id]
            row = _metrics(y_valid, scores, threshold, scenario)
            row.update({
                "fold": fold, "scenario": scenario.name, "cost_family": scenario.family,
                "selection_scope": scope, "selected_candidate": candidate_id,
                "selected_family": selected["candidate_family"],
                "params_json": selected["params_json"],
                "inner_selection_cost_per_product": selected["cost_per_product"],
                "estimate_type": "nested_outer_evaluation",
            })
            metric_rows.append(row)
            prediction_frames.append(pd.DataFrame({
                "scenario": scenario.name, "cost_family": scenario.family,
                "selection_scope": scope, "fold": fold, "row_index": X_valid.index,
                "y_true": y_valid, "y_score": scores, "y_pred": (scores >= threshold).astype(int),
                "threshold": threshold, "selected_candidate": candidate_id,
            }))
        for scenario in scenarios:
            for scope, score, threshold in (("always_pass", 0.0, 1.0), ("review_all", 1.0, 0.0)):
                scores = np.full(len(y_valid), score)
                row = _metrics(y_valid, scores, threshold, scenario)
                row.update({
                    "fold": fold, "scenario": scenario.name, "cost_family": scenario.family,
                    "selection_scope": scope, "selected_candidate": scope,
                    "selected_family": "baseline", "params_json": "{}",
                    "estimate_type": "nested_outer_evaluation",
                })
                metric_rows.append(row)
                prediction_frames.append(pd.DataFrame({
                    "scenario": scenario.name, "cost_family": scenario.family,
                    "selection_scope": scope, "fold": fold, "row_index": X_valid.index,
                    "y_true": y_valid, "y_score": scores,
                    "y_pred": (scores >= threshold).astype(int), "threshold": threshold,
                    "selected_candidate": scope,
                }))
    fold_metrics = pd.DataFrame(metric_rows)
    return {
        "fold_metrics": fold_metrics, "summary": _summarize(fold_metrics),
        "outer_predictions": pd.concat(prediction_frames, ignore_index=True),
        "inner_search": pd.concat(searches, ignore_index=True),
    }


def fit_final_cost_policies(
    X: pd.DataFrame, y: Any, candidates: Sequence[CandidateSpec],
    scenarios: Sequence[CostScenario], cv_splits: int = 5,
    seed: int = 42, verbose: bool = True,
) -> dict[str, Any]:
    """Select/refit on all training data; report selection-only OOF statistics."""
    labels = _validate_data(X, y, cv_splits)
    _validate_search(candidates, scenarios)
    search = _selection_search(X, labels, candidates, scenarios, cv_splits, seed, verbose)
    by_id = {c.candidate_id: c for c in candidates}
    policies, models = [], {}
    for scenario, scope, selected in _selected_rows(search, scenarios):
        candidate_id = selected["candidate_id"]
        if candidate_id not in models:
            model = build_pipeline(by_id[candidate_id], seed)
            with threadpool_limits(limits=1):
                model.fit(X, labels)
            models[candidate_id] = model
        row = dict(selected)
        row.update({
            "selection_scope": scope, "selected_candidate": candidate_id,
            "selected_family": selected["candidate_family"],
            "estimate_type": "selection_only_not_unbiased",
        })
        policies.append(row)
    return {"policies": pd.DataFrame(policies), "selection_search": search, "models": models}


def save_final_cost_policies(
    result: dict[str, Any], output_dir: str | Path,
    scenarios: Sequence[CostScenario], metadata: dict[str, Any] | None = None,
) -> Path:
    """Save only in this experiment directory, never replace original artifacts."""
    destination = Path(output_dir)
    model_dir = destination / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for candidate_id, model in result["models"].items():
        joblib.dump(model, model_dir / f"{candidate_id}.joblib")
    result["policies"].to_csv(destination / "final_policies.csv", index=False, encoding="utf-8-sig")
    result["selection_search"].to_csv(
        destination / "final_selection_search.csv", index=False, encoding="utf-8-sig"
    )
    policies = []
    for row in result["policies"].to_dict(orient="records"):
        policies.append({
            "scenario": row["scenario"], "cost_family": row["cost_family"],
            "selection_scope": row["selection_scope"],
            "selected_family": row["selected_family"],
            "selected_candidate": row["selected_candidate"],
            "threshold": float(row["threshold"]), "params": json.loads(row["params_json"]),
            "model_path": f"models/{row['selected_candidate']}.joblib",
        })
    config = {
        "costs_are_assumptions": True, "cost_unit": "relative_unit_not_currency",
        "timestamp_used_as_feature": False, "holdout_used_for_tuning_or_evaluation": False,
        "score_is_calibrated_probability": False,
        "selection_statistics_are_unbiased": False,
        "cost_formula": "(tp_cost*TP + fp_cost*FP + fn_cost*FN + tn_cost*TN) / N",
        "tie_rule": "lowest cost, fewest inspections, highest threshold, candidate ID",
        "review_assumption": "Every flagged product is reviewed; review detects Fail. Miss cost is expected residual cost at this decision step.",
        "scenarios": [asdict(s) for s in scenarios], "policies": policies,
        "metadata": metadata or {},
    }
    path = destination / "policy_configs.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return path

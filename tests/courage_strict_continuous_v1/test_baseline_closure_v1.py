from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "examples/courage_strict_continuous_v1/evaluate_baseline_closure_v1.py"
SPEC = importlib.util.spec_from_file_location("baseline_closure_v1", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
DIAGNOSTIC_CONTRACT = (
    ROOT / "examples/courage_strict_continuous_v1/baseline_closure_diagnostic_v1.json"
)


def test_direction_metrics_balanced_case() -> None:
    target = np.array([1.0, 1.0, -1.0, -1.0])
    prediction = np.array([1.0, -1.0, -1.0, 1.0])
    metrics = MODULE.direction_metrics(target, prediction)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["mcc"] == pytest.approx(0.0)
    assert metrics["auc"] == pytest.approx(0.5)


def test_ridge_keeps_intercept_unpenalized() -> None:
    design = np.column_stack([np.ones(5), np.arange(5, dtype=np.float64)])
    target = np.full(5, 3.0)
    coefficient = MODULE.solve_ridge(
        design.T @ design,
        design.T @ target,
        rows=5,
        alpha_on_average_gram=1.0,
    )
    assert coefficient[0] == pytest.approx(3.0)
    assert coefficient[1] == pytest.approx(0.0, abs=1e-12)


def test_gate_requires_mean_and_four_horizons() -> None:
    rows = [
        {
            "horizon": horizon,
            "estimator": "model",
            "rmse_skill_vs_best_baseline": skill,
        }
        for horizon, skill in zip(
            MODULE.HORIZONS_V1,
            [0.01, 0.01, 0.01, 0.01, -0.001, -0.001, -0.001],
            strict=True,
        )
    ]
    assert MODULE.gate_decision(rows)["decision"] == "PASS_BASELINE_GATE"
    rows[0]["rmse_skill_vs_best_baseline"] = -0.1
    assert MODULE.gate_decision(rows)["decision"] == "FAIL_BASELINE_GATE"


def test_diagnostic_contract_binds_versioned_source_experiment() -> None:
    contract, source = MODULE._validate_contract(DIAGNOSTIC_CONTRACT)
    assert contract["source_experiment_id"] == source["experiment_id"]
    assert contract["checkpoint_step"] == 2250
    assert contract["authority"]["april_or_later_read"] is False

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "examples/courage_strict_continuous_v1/run_sanity_checks_v1.py"
SPEC = importlib.util.spec_from_file_location("run_sanity_checks_v1", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_random_control_metrics_are_random_for_independent_scores() -> None:
    rng = np.random.default_rng(7)
    rows = 100_000
    calendar = np.repeat(np.arange(1000), rows // 1000)
    target = rng.normal(size=rows)
    prediction = rng.normal(size=rows)
    result = MODULE._random_control_metrics(calendar, target, prediction)
    assert abs(result["balanced_accuracy"] - 0.5) < 0.01
    assert abs(result["auc"] - 0.5) < 0.01
    assert abs(result["mcc"]) < 0.02
    assert abs(result["rank_ic_mean"]) < 0.02


def test_time_stratified_choice_is_unique_and_reproducible() -> None:
    raw = type("Raw", (), {"ends": np.arange(2000, dtype=np.int64)})()
    indices = np.arange(2000, dtype=np.int64)
    first = MODULE._time_stratified_choice(raw, indices, 100, 11)
    second = MODULE._time_stratified_choice(raw, indices, 100, 11)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 100

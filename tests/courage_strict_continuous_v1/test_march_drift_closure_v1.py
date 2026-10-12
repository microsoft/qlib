from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "examples/courage_strict_continuous_v1/evaluate_march_drift_closure_v1.py"
SPEC = importlib.util.spec_from_file_location("march_drift_closure_v1", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_contract_pins_frozen_pre_april_inputs() -> None:
    contract = MODULE.validate_contract()
    assert contract["source_experiment_id"] == "courage_strict_continuous_v1"
    assert contract["segments"]["march"] == ["2026-03-02", "2026-04-01"]
    assert contract["scope"] == {
        "read_before_exclusive": "2026-04-01",
        "april_or_later_read": False,
    }
    assert contract["authority"]["train_model"] is False
    assert contract["authority"]["remote_push"] is False


@pytest.mark.parametrize("rows,maximum,expected", [(10, 20, 10), (100, 17, 17)])
def test_systematic_indices_are_unique_and_in_range(
    rows: int, maximum: int, expected: int
) -> None:
    indices = MODULE._systematic_indices(rows, maximum)
    assert len(indices) == expected
    assert len(np.unique(indices)) == expected
    assert indices[0] >= 0
    assert indices[-1] < rows


def test_direction_majority_excludes_zero_returns() -> None:
    target = np.array([-1.0, -0.5, 0.0, 1.0])
    assert MODULE._direction_majority(target) == pytest.approx(2 / 3)

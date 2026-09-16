from __future__ import annotations

import numpy as np
import pytest

from qlib.contrib.data.courage_strict_v1.evaluate import _cross_sectional, _direction


def test_direction_metrics_exclude_flat_targets_without_calling_them_down() -> None:
    result = _direction(
        np.array([1.0, -1.0, 0.0, 2.0, -2.0]),
        np.array([1.0, -1.0, -1.0, -1.0, 1.0]),
    )
    assert result["active_rows"] == 4
    assert result["flat_target_rows"] == 1
    assert result["accuracy"] == 0.5
    assert result["balanced_accuracy"] == 0.5
    assert result["target_up_rate"] == 0.5


def test_direction_metrics_report_majority_baseline() -> None:
    result = _direction(
        np.array([-1.0, -2.0, -3.0, 1.0]),
        np.array([-1.0, -1.0, -1.0, -1.0]),
    )
    assert result["accuracy"] == 0.75
    assert result["majority_accuracy"] == 0.75
    assert result["balanced_accuracy"] == 0.5


def test_cross_sectional_metrics_keep_rank_and_spread_separate() -> None:
    timestamp = np.repeat(np.array([1, 2]), 10)
    target = np.tile(np.arange(10, dtype=float), 2)
    prediction = target.copy()
    result = _cross_sectional(timestamp, target, prediction)
    assert result["rank_ic_mean"] == pytest.approx(1.0)
    assert result["rank_ic_timestamps"] == 2
    assert result["top_bottom_spread_mean"] == pytest.approx(8.5)

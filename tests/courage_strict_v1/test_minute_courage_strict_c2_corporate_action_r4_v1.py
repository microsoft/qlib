from __future__ import annotations

import pandas as pd
import pytest

from qlib.contrib.data.courage_strict_v1.minute_courage_strict_c2_corporate_action_r4_v1 import (
    CourageStrictC2CorporateActionR4Error,
    build_quarantine_v1,
    validate_partition_v1,
)


def _factor() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["A.SH", "B.SZ"],
            "trade_date": pd.to_datetime(["2025-04-08", "2025-04-09"]),
            "event_time": pd.to_datetime(["2025-04-08", "2025-04-09"], utc=True),
            "adj_factor": [1.1, 2.1],
            "previous_adjustment_factor": [1.0, 2.0],
            "factor_ratio": [1.1, 1.05],
        }
    )


def test_quarantine_is_fail_closed() -> None:
    result = build_quarantine_v1(_factor().iloc[[1]])
    assert result.loc[0, "instrument"] == "B.SZ"
    assert not result.loc[0, "adjustment_factor_use_authorized"]
    assert not result.loc[0, "canonical_action_inference_authorized"]
    assert not result.loc[0, "label_head_crossing_event_valid"]
    assert result.loc[0, "feature_history_reset_required"]
    assert (
        result.loc[0, "minimum_post_event_official_minutes_before_dynamic_use"] == 1200
    )


def test_partition_is_exhaustive_and_disjoint() -> None:
    actions = pd.DataFrame(
        {"instrument": ["A.SH"], "ex_date": [pd.Timestamp("2025-04-08")]}
    )
    quarantine = build_quarantine_v1(_factor().iloc[[1]])
    coverage = validate_partition_v1(
        factor_changes=_factor(), actions=actions, quarantine=quarantine
    )
    assert coverage == {
        "factor_change_rows": 2,
        "accepted_action_rows": 1,
        "quarantined_factor_change_rows": 1,
        "classified_rows": 2,
    }


def test_partition_rejects_unclassified_change() -> None:
    actions = pd.DataFrame(
        {"instrument": ["A.SH"], "ex_date": [pd.Timestamp("2025-04-08")]}
    )
    with pytest.raises(CourageStrictC2CorporateActionR4Error, match="not exhaustive"):
        validate_partition_v1(
            factor_changes=_factor(),
            actions=actions,
            quarantine=build_quarantine_v1(_factor().iloc[0:0]),
        )

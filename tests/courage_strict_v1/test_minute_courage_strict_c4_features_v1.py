from __future__ import annotations

import numpy as np
import pandas as pd

from qlib.contrib.data.courage_strict_v1.minute_courage_strict_c4_features_v1 import (
    DYNAMIC_FEATURES,
    SLOW_FEATURES,
    build_slow_context_v1,
    compute_sector_features_day_v1,
    compute_stock_features_v1,
)


def test_stock_features_are_causal_and_event_reset() -> None:
    length = 1305
    close = 10.0 + np.arange(length) * 0.001
    high = close + 0.01
    low = close - 0.01
    volume = np.full(length, 100.0)
    amount = close * volume
    values, available, missing = compute_stock_features_v1(
        close=close,
        high=high,
        low=low,
        volume=volume,
        amount=amount,
        event_offsets=np.array([30]),
        quarantine_offsets=np.array([60]),
    )
    assert values.shape == (length, 8)
    assert not available[0, 0]
    assert available[1, 0] and not missing[1, 0]
    assert available[20, 7] and not missing[20, 7]
    assert not available[30, 0]
    assert available[31, 0]
    assert not available[60:1260].any()
    assert available[1260, 0]
    assert np.isfinite(values[available & ~missing]).all()


def test_missing_bar_is_explicit_data_missing_not_padding() -> None:
    length = 50
    close = np.full(length, 10.0)
    high = np.full(length, 10.1)
    low = np.full(length, 9.9)
    volume = np.full(length, 100.0)
    amount = np.full(length, 1000.0)
    volume[25] = 0.0
    amount[25] = 0.0
    _, available, missing = compute_stock_features_v1(
        close=close,
        high=high,
        low=low,
        volume=volume,
        amount=amount,
        event_offsets=np.array([], dtype=np.int64),
        quarantine_offsets=np.array([], dtype=np.int64),
    )
    assert available[25].all()
    assert missing[25].all()
    assert missing[26, 0]
    assert missing[30, 1]


def test_sector_gate_uses_complete_sector_denominator() -> None:
    values = np.zeros((4, 240, 8), dtype=np.float32)
    values[:, :, 0] = np.array([0.01, -0.01, 0.02, -0.02])[:, None]
    values[:, :, 1] = np.array([0.05, 0.01, 0.03, -0.01])[:, None]
    values[:, :, 2] = np.array([0.10, 0.02, 0.06, -0.02])[:, None]
    values[:, :, 7] = np.array([1.0, 2.0, 3.0, 4.0])[:, None]
    available = np.ones_like(values, dtype=bool)
    missing = np.zeros_like(values, dtype=bool)
    result = compute_sector_features_day_v1(
        stock_values=values,
        stock_available=available,
        stock_missing=missing,
        sector_codes=np.array(["S1"] * 4, dtype=object),
        active_symbols=np.ones(4, dtype=bool),
        minimum_coverage=0.80,
        minimum_valid_count=3,
    )
    output, output_available, output_missing, counts = result["S1"]
    assert np.isclose(output[0, 0], 0.02)
    assert np.isclose(output[0, 1], 0.04)
    assert np.isclose(output[0, 2], 0.5)
    assert np.isclose(output[0, 3], 2.5)
    assert output_available.all() and not output_missing.any()
    assert (counts == 4).all()

    missing[0, 0, 1] = True
    result = compute_sector_features_day_v1(
        stock_values=values,
        stock_available=available,
        stock_missing=missing,
        sector_codes=np.array(["S1"] * 4, dtype=object),
        active_symbols=np.ones(4, dtype=bool),
        minimum_coverage=0.80,
        minimum_valid_count=3,
    )
    _, output_available, output_missing, _ = result["S1"]
    assert output_available[0, 0]
    assert output_missing[0, 0]


def test_slow_context_is_strict_t_minus_one() -> None:
    dates = pd.bdate_range("2025-04-01", periods=70)
    daily = pd.DataFrame(
        {
            "instrument": "600000.SH",
            "session_date": dates,
            "close_cny": 10.0 + np.arange(70) * 0.1,
            "total_mv_10k_cny": 100000.0,
            "strict_t_minus_1_source_use_authorized": True,
            "same_session_intraday_use_authorized": False,
        }
    )
    membership = pd.DataFrame(
        {
            "instrument": ["600000.SH"],
            "signal_date": [dates[-1]],
            "turnover_mean_60_percent": [8.0],
            "strict_T_minus_1": [True],
        }
    )
    result = build_slow_context_v1(
        membership=membership, daily=daily, official_dates=dates
    )
    assert result.iloc[0]["slow_source_date"] == dates[-2]
    assert all(feature in result for feature in SLOW_FEATURES)
    assert not result.filter(like="__data_missing").iloc[0].any()
    assert len(DYNAMIC_FEATURES) == 12


def test_slow_context_accepts_signal_day_immediately_after_history() -> None:
    dates = pd.bdate_range("2025-04-01", periods=70)
    daily = pd.DataFrame(
        {
            "instrument": "600000.SH",
            "session_date": dates,
            "close_cny": 10.0 + np.arange(70) * 0.1,
            "total_mv_10k_cny": 100000.0,
            "strict_t_minus_1_source_use_authorized": True,
            "same_session_intraday_use_authorized": False,
        }
    )
    signal_date = dates[-1] + pd.offsets.BDay(1)
    membership = pd.DataFrame(
        {
            "instrument": ["600000.SH"],
            "signal_date": [signal_date],
            "turnover_mean_60_percent": [8.0],
            "strict_T_minus_1": [True],
        }
    )
    result = build_slow_context_v1(
        membership=membership, daily=daily, official_dates=dates
    )
    assert result.iloc[0]["slow_source_date"] == dates[-1]
    assert not result.filter(like="__data_missing").iloc[0].any()

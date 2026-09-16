from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlib.contrib.data.courage_strict_v1.minute_courage_strict_c3_labels_v1 import (
    HORIZONS,
    STATUS_ACTION_CROSSING,
    STATUS_ENTRY_LIMIT_LOCKED,
    STATUS_TARGET_MISSING,
    CourageStrictC3LabelError,
    _validate_axis,
    build_symbol_sample_partition_v1,
)


def _axis(days: int = 4) -> pd.DataFrame:
    dates = pd.bdate_range("2025-07-01", periods=days)
    rows: list[dict[str, object]] = []
    for date in dates:
        slot = 0
        for start, count in (("09:30", 120), ("13:00", 120)):
            base = pd.Timestamp(f"{date.date()} {start}").tz_localize("Asia/Shanghai")
            for minute in range(count):
                window_start = base + pd.Timedelta(minutes=minute)
                rows.append(
                    {
                        "session_date": date,
                        "exchange": "SSE",
                        "minute_slot": slot,
                        "minute_window_start": window_start,
                        "minute_window_end": window_start + pd.Timedelta(minutes=1),
                        "feature_ready_time": window_start + pd.Timedelta(minutes=2),
                        "is_official_open_minute": True,
                    }
                )
                slot += 1
    return _validate_axis(pd.DataFrame(rows))


def _inputs() -> tuple[pd.DataFrame, ...]:
    axis = _axis()
    signal_date = pd.Timestamp("2025-07-01")
    membership = pd.DataFrame(
        {
            "instrument": ["600000.SH"],
            "signal_date": [signal_date],
            "role": ["train"],
            "turnover_mean_60_percent": [8.0],
            "turnover_valid_observations": np.array([60], dtype=np.int16),
            "turnover_window_start": [pd.Timestamp("2025-04-01")],
            "turnover_window_end": [pd.Timestamp("2025-06-30")],
            "status_source_date": [pd.Timestamp("2025-06-30")],
            "listing_official_sessions_before_signal": np.array([200], dtype=np.int32),
            "membership_lower_inclusive": [5.0],
            "membership_upper_inclusive": [15.0],
            "strict_T_minus_1": [True],
        }
    )
    chosen = axis.iloc[np.arange(0, 240, 5)].copy()
    grid = pd.DataFrame(
        {
            "session_date": chosen["session_date"].to_numpy(),
            "signal_time": chosen["feature_ready_time"].to_numpy(),
            "minute_slot": chosen["minute_slot"].to_numpy(dtype=np.int16),
            "global_minute_offset": chosen["global_minute_offset"].to_numpy(),
            "legal_state_index": np.arange(0, 240, 5, dtype=np.int16),
            "entry_global_minute_offset": chosen["global_minute_offset"].to_numpy() + 2,
        }
    )
    bars = pd.DataFrame(
        {
            "ts_code": "600000.SH",
            "trade_time": axis["minute_window_end"].dt.tz_localize(None),
            "open": 10.0,
            "high": 10.1,
            "low": 9.9,
            "close": 10.0,
            "vol": 100.0,
            "amount": 1000.0 + np.arange(len(axis), dtype=float),
        }
    )
    dates = pd.DatetimeIndex(axis["session_date"].unique())
    status = pd.DataFrame(
        {
            "session_date": dates,
            "is_suspended": False,
            "limit_up_price": 11.0,
            "limit_down_price": 9.0,
        }
    )
    industry = pd.DataFrame(
        {
            "session_date": [signal_date],
            "industry_code": ["I1"],
            "sector_level2_code": ["S1"],
            "industry_known": [True],
            "change_date": [pd.Timestamp("2020-01-01")],
        }
    )
    return membership, grid, axis, bars, industry, status


def _build(*, action_dates: np.ndarray | None = None) -> pd.DataFrame:
    membership, grid, axis, bars, industry, status = _inputs()
    frame, evidence = build_symbol_sample_partition_v1(
        instrument="600000.SH",
        membership=membership,
        grid=grid,
        axis=axis,
        bars=bars,
        industry=industry,
        daily_status=status,
        action_dates=(
            np.array([], dtype="datetime64[ns]")
            if action_dates is None
            else action_dates
        ),
        role_cutoffs={"train": pd.Timestamp("2025-07-08", tz="Asia/Shanghai")},
    )
    assert evidence["rows"] == 48
    return frame


def test_all_heads_and_sample_identity_are_exact() -> None:
    frame = _build()
    assert len(frame) == 48
    assert frame["sample_id"].is_unique
    assert all(f"gross_return_vwap1_{horizon}m" in frame for horizon in HORIZONS)
    assert all(
        frame[f"objective_gross_valid_vwap1_{horizon}m"].any() for horizon in HORIZONS
    )


def test_missing_target_invalidates_only_affected_head() -> None:
    membership, grid, axis, bars, industry, status = _inputs()
    first_entry = int(grid.iloc[0]["entry_global_minute_offset"])
    missing_end = axis.iloc[first_entry + 5]["minute_window_end"].tz_localize(None)
    bars = bars.loc[bars["trade_time"].ne(missing_end)]
    frame, _ = build_symbol_sample_partition_v1(
        instrument="600000.SH",
        membership=membership,
        grid=grid,
        axis=axis,
        bars=bars,
        industry=industry,
        daily_status=status,
        action_dates=np.array([], dtype="datetime64[ns]"),
        role_cutoffs={"train": pd.Timestamp("2025-07-08", tz="Asia/Shanghai")},
    )
    assert frame.iloc[0]["label_status_vwap1_5m"] == STATUS_TARGET_MISSING
    assert not bool(frame.iloc[0]["objective_gross_valid_vwap1_5m"])
    assert bool(frame.iloc[0]["objective_gross_valid_vwap1_15m"])


def test_action_crossing_and_limit_lock_fail_closed() -> None:
    frame = _build(action_dates=np.array(["2025-07-03"], dtype="datetime64[ns]"))
    crossing = frame["label_status_vwap1_480m"].eq(STATUS_ACTION_CROSSING)
    assert crossing.any()
    assert not frame.loc[crossing, "objective_gross_valid_vwap1_480m"].any()

    membership, grid, axis, bars, industry, status = _inputs()
    entry_end = axis.iloc[int(grid.iloc[0]["entry_global_minute_offset"])][
        "minute_window_end"
    ].tz_localize(None)
    mask = bars["trade_time"].eq(entry_end)
    bars.loc[mask, ["open", "high", "low", "close"]] = 11.0
    frame, _ = build_symbol_sample_partition_v1(
        instrument="600000.SH",
        membership=membership,
        grid=grid,
        axis=axis,
        bars=bars,
        industry=industry,
        daily_status=status,
        action_dates=np.array([], dtype="datetime64[ns]"),
        role_cutoffs={"train": pd.Timestamp("2025-07-08", tz="Asia/Shanghai")},
    )
    assert frame.iloc[0]["label_status_vwap1_5m"] == STATUS_ENTRY_LIMIT_LOCKED
    assert not bool(frame.iloc[0]["objective_gross_valid_vwap1_5m"])


def test_missing_same_day_industry_is_retained_as_unknown() -> None:
    membership, grid, axis, bars, industry, status = _inputs()
    industry["session_date"] = pd.Timestamp("2025-07-02")
    frame, evidence = build_symbol_sample_partition_v1(
        instrument="600000.SH",
        membership=membership,
        grid=grid,
        axis=axis,
        bars=bars,
        industry=industry,
        daily_status=status,
        action_dates=np.array([], dtype="datetime64[ns]"),
        role_cutoffs={"train": pd.Timestamp("2025-07-08", tz="Asia/Shanghai")},
    )
    assert evidence["unknown_industry_rows"] == 48
    assert not frame["industry_known"].any()
    assert frame["industry_code"].isna().all()


def test_axis_rejects_non_240_session() -> None:
    axis = _axis().iloc[:-1]
    with pytest.raises(CourageStrictC3LabelError, match="240 slots"):
        _validate_axis(axis)

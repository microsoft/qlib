from __future__ import annotations

import pandas as pd

from qlib.contrib.data.courage_strict_v1.minute_courage_strict_c2_daily_r5_v1 import (
    normalize_daily_records_v1,
)


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["A.SH", "B.SZ"],
            "trade_date": pd.to_datetime(["2025-04-01", "2025-04-01"]),
            "close": [10.0, 20.0],
            "vol": [100.0, 200.0],
            "turnover_rate": [0.01, 0.02],
            "turnover_rate_f": [0.02, 2.0],
            "total_share": [1000.0, 2000.0],
            "float_share": [10000.0, 10000.0],
            "free_share": [5000.0, 10000.0],
            "total_mv": [10000.0, 40000.0],
            "circ_mv": [100000.0, 200000.0],
        }
    )


def _ready() -> pd.Series:
    return pd.Series(
        [pd.Timestamp("2025-04-01 15:01", tz="Asia/Shanghai")],
        index=[pd.Timestamp("2025-04-01")],
    )


def test_normalization_quarantines_only_bad_field_and_forbids_same_day() -> None:
    normalized, quarantine = normalize_daily_records_v1(
        _source(), final_ready_by_date=_ready()
    )
    assert len(normalized) == 2
    assert normalized["daily_available_at"].dt.hour.eq(15).all()
    assert normalized["daily_available_at"].dt.minute.eq(1).all()
    assert normalized["strict_t_minus_1_source_use_authorized"].all()
    assert not normalized["same_session_intraday_use_authorized"].any()
    assert normalized.loc[0, "turnover_rate_f_unit_valid"]
    assert not normalized.loc[1, "turnover_rate_f_unit_valid"]
    assert pd.isna(normalized.loc[1, "turnover_rate_f_percent"])
    assert quarantine[["instrument", "field"]].to_records(index=False).tolist() == [
        ("B.SZ", "turnover_rate_f")
    ]
    assert not quarantine["field_use_authorized"].any()


def test_turnover_rounding_uses_absolute_tolerance() -> None:
    source = _source().iloc[[0]].copy()
    source.loc[:, "turnover_rate"] = 0.0109
    normalized, quarantine = normalize_daily_records_v1(
        source, final_ready_by_date=_ready()
    )
    assert normalized.loc[0, "turnover_rate_unit_valid"]
    assert quarantine.empty

from __future__ import annotations

import pandas as pd

from qlib.contrib.data.courage_strict_v1.minute_courage_strict_c3_membership_v1 import (
    build_membership_v1,
)


def test_membership_uses_strict_prior_60_sessions() -> None:
    dates = pd.bdate_range("2024-01-01", periods=181)
    scope = pd.DataFrame(
        {"instrument": ["A.SH"], "list_date": [dates[0]], "delist_date": [pd.NaT]}
    )
    daily = pd.DataFrame(
        {
            "instrument": "A.SH",
            "session_date": dates,
            "turnover_rate_f_percent": 10.0,
        }
    )
    # Current-day outlier must not affect current-day membership.
    daily.loc[daily["session_date"].eq(dates[-1]), "turnover_rate_f_percent"] = 1000.0
    status = pd.DataFrame(
        {
            "instrument": "A.SH",
            "session_date": dates,
            "membership_status_fail_closed_excluded": False,
        }
    )
    result = build_membership_v1(
        scope=scope,
        daily=daily,
        status=status,
        full_open_dates=dates.to_numpy(dtype="datetime64[D]"),
        signal_dates=pd.DatetimeIndex([dates[-1]]),
        role_by_date={dates[-1]: "train"},
    )
    assert len(result) == 1
    assert result.loc[0, "turnover_mean_60_percent"] == 10.0
    assert result.loc[0, "status_source_date"] == dates[-2]

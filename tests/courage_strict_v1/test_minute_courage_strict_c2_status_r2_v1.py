from __future__ import annotations

import pandas as pd
import pytest

from qlib.contrib.data.courage_strict_v1.minute_courage_strict_c2_status_r2_v1 import (
    CourageStrictC2StatusR2Error,
    TerminationEventV1,
    build_status_table_v1,
    classify_termination_events_v1,
)


def _daily() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2025-04-01"] * 3 + ["2025-04-02"] * 3),
            "ts_code": ["000001.SZ", "000002.SZ", "600001.SH"] * 2,
            "is_st": [0, 1, 0, 0, 1, 0],
            "suspend_type": ["N", "N", "S", "N", "R", "N"],
            "up_limit": [11.0, 5.5, 22.0, 11.2, 5.6, 22.2],
            "down_limit": [9.0, 4.5, 18.0, 9.2, 4.6, 18.2],
        }
    )


def _ready() -> pd.Series:
    values = pd.to_datetime(["2025-04-01 15:01:00+08:00", "2025-04-02 15:01:00+08:00"])
    return pd.Series(values, index=pd.to_datetime(["2025-04-01", "2025-04-02"]))


def test_status_risk_is_nullable_and_unknown_fails_closed() -> None:
    frame = build_status_table_v1(
        daily=_daily(), session_ready=_ready(), termination_events=[]
    )
    st = frame.loc[frame["instrument"].eq("000002.SZ")]
    assert st["is_ST"].all()
    assert st["has_delisting_risk"].isna().all()
    assert st["membership_status_fail_closed_excluded"].all()
    ordinary = frame.loc[frame["instrument"].eq("000001.SZ")]
    assert ordinary["has_delisting_risk"].eq(False).all()
    assert ordinary["delisting_risk_known"].all()


def test_affirmative_termination_event_turns_risk_true_only_after_available() -> None:
    event = TerminationEventV1(
        instrument="600001.SH",
        announcement_time=pd.Timestamp("2025-04-01 20:00:00", tz="Asia/Shanghai"),
        announcement_id="a",
        title="关于收到股票终止上市决定的公告",
    )
    frame = build_status_table_v1(
        daily=_daily(), session_ready=_ready(), termination_events=[event]
    )
    selected = frame.loc[frame["instrument"].eq("600001.SH")]
    assert selected["has_delisting_risk"].tolist() == [False, True]
    assert selected["membership_status_fail_closed_excluded"].tolist() == [False, True]


def test_termination_classifier_rejects_risk_hint_and_bond_notice() -> None:
    items = [
        {
            "secCode": "000001",
            "announcementTime": 1_743_600_000_000,
            "announcementId": "good",
            "announcementTitle": "关于收到股票<em>终止上市</em>决定的公告",
        },
        {
            "secCode": "000001",
            "announcementTime": 1_743_600_000_001,
            "announcementId": "hint",
            "announcementTitle": "股票可能被终止上市的风险提示公告",
        },
        {
            "secCode": "000001",
            "announcementTime": 1_743_600_000_002,
            "announcementId": "bond",
            "announcementTitle": "公司债券终止上市的公告",
        },
    ]
    events = classify_termination_events_v1(items, admitted_instruments={"000001.SZ"})
    assert [event.announcement_id for event in events] == ["good"]


def test_status_rejects_duplicate_keys() -> None:
    daily = pd.concat([_daily(), _daily().iloc[[0]]], ignore_index=True)
    with pytest.raises(CourageStrictC2StatusR2Error, match="duplicate"):
        build_status_table_v1(
            daily=daily, session_ready=_ready(), termination_events=[]
        )


def test_same_session_intraday_use_is_always_false() -> None:
    frame = build_status_table_v1(
        daily=_daily(), session_ready=_ready(), termination_events=[]
    )
    assert not frame["same_session_intraday_use_authorized"].any()
    assert frame["status_available_at"].dt.hour.eq(15).all()
    assert frame["status_available_at"].dt.minute.eq(1).all()

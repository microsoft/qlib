from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qlib.contrib.data.courage_strict_v1.minute_courage_strict_c2_official_axis_audit_v1 import (
    CourageStrictOfficialAxisAuditError,
    audit_official_minute_axis_v1,
    expected_clock_labels_sha256_v1,
)
from qlib.contrib.data.courage_strict_v1.minute_courage_strict_c2_official_axis_v1 import (
    build_official_minute_axis_v1,
)


def _config() -> dict:
    return {
        "record_read_interval": ["2025-04-01", "2025-04-03"],
        "expected_open_sessions": 1,
        "exchanges": ["SSE", "SZSE"],
        "timezone": "Asia/Shanghai",
        "session_rule": {
            "rule_version": "CN_A_SHARE_AUCTION_SESSIONS_2023_V1",
            "intervals": [
                {"name": "morning", "start": "09:30", "minutes": 120},
                {"name": "afternoon", "start": "13:00", "minutes": 120},
            ],
        },
    }


def _calendar(path: Path) -> None:
    table = pa.table(
        {
            "trade_date": pa.array(
                [pd.Timestamp("2025-04-01"), pd.Timestamp("2025-04-02")],
                type=pa.timestamp("ns"),
            ),
            "exchange": ["SSE_REFERENCE", "SSE_REFERENCE"],
            "is_open": pa.array([1, 0], type=pa.int8()),
        }
    )
    pq.write_table(table, path)


def _write_source(path: Path) -> list[dict[str, str]]:
    path.write_bytes(b"official rule snapshot")
    import hashlib

    return [
        {
            "path": path.as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    ]


def test_build_and_independent_audit_pass(tmp_path: Path) -> None:
    calendar = tmp_path / "calendar.parquet"
    _calendar(calendar)
    axis = tmp_path / "axis.parquet"
    pq.write_table(
        build_official_minute_axis_v1(calendar_path=calendar, config=_config()), axis
    )
    result = audit_official_minute_axis_v1(
        axis_path=axis,
        calendar_path=calendar,
        source_records=_write_source(tmp_path / "rule.bin"),
        interval=("2025-04-01", "2025-04-03"),
        expected_open_sessions=1,
        expected_clock_labels_sha256=expected_clock_labels_sha256_v1(),
    )
    assert result["rows"] == 480
    assert result["lunch_slots_emitted"] == 0
    frame = pq.read_table(axis).to_pandas()
    sse = frame[frame["exchange"].eq("SSE")]
    assert sse.iloc[0]["minute_window_start"].strftime("%H:%M") == "09:30"
    assert sse.iloc[119]["minute_window_end"].strftime("%H:%M") == "11:30"
    assert sse.iloc[120]["minute_window_start"].strftime("%H:%M") == "13:00"
    assert sse.iloc[-1]["minute_window_end"].strftime("%H:%M") == "15:00"


@pytest.mark.parametrize(
    ("column", "delta"),
    [
        ("minute_window_end", pd.Timedelta(minutes=1)),
        ("feature_ready_time", pd.Timedelta(minutes=1)),
    ],
)
def test_independent_audit_rejects_clock_drift(
    tmp_path: Path, column: str, delta: pd.Timedelta
) -> None:
    calendar = tmp_path / "calendar.parquet"
    _calendar(calendar)
    table = build_official_minute_axis_v1(calendar_path=calendar, config=_config())
    frame = table.to_pandas()
    frame.loc[0, column] += delta
    axis = tmp_path / "axis.parquet"
    pq.write_table(
        pa.Table.from_pandas(frame, schema=table.schema, preserve_index=False), axis
    )
    with pytest.raises(
        CourageStrictOfficialAxisAuditError, match="clock semantic drift"
    ):
        audit_official_minute_axis_v1(
            axis_path=axis,
            calendar_path=calendar,
            source_records=_write_source(tmp_path / "rule.bin"),
            interval=("2025-04-01", "2025-04-03"),
            expected_open_sessions=1,
            expected_clock_labels_sha256=expected_clock_labels_sha256_v1(),
        )


def test_independent_audit_rejects_source_drift(tmp_path: Path) -> None:
    calendar = tmp_path / "calendar.parquet"
    _calendar(calendar)
    axis = tmp_path / "axis.parquet"
    pq.write_table(
        build_official_minute_axis_v1(calendar_path=calendar, config=_config()), axis
    )
    source = tmp_path / "rule.bin"
    records = _write_source(source)
    source.write_bytes(b"changed")
    with pytest.raises(CourageStrictOfficialAxisAuditError, match="source byte drift"):
        audit_official_minute_axis_v1(
            axis_path=axis,
            calendar_path=calendar,
            source_records=records,
            interval=("2025-04-01", "2025-04-03"),
            expected_open_sessions=1,
            expected_clock_labels_sha256=expected_clock_labels_sha256_v1(),
        )

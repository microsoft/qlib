"""Independent semantic audit for the Courage strict C2 official-minute axis.

This module deliberately does not import the axis materializer.  The audit is
an independent implementation of the frozen clock semantics so a generator
bug cannot pass merely because generation and validation share constants.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Final

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class CourageStrictOfficialAxisAuditError(RuntimeError):
    """Raised when official-minute-axis evidence is incomplete or inconsistent."""


TIMEZONE: Final[str] = "Asia/Shanghai"
EXPECTED_EXCHANGES: Final[tuple[str, ...]] = ("SSE", "SZSE")
EXPECTED_RULE_VERSION: Final[str] = "CN_A_SHARE_AUCTION_SESSIONS_2023_V1"
EXPECTED_COLUMNS: Final[tuple[str, ...]] = (
    "session_date",
    "exchange",
    "minute_slot",
    "session",
    "minute_in_session",
    "minute_window_start",
    "minute_window_end",
    "feature_ready_time",
    "is_official_open_minute",
    "rule_version",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expected_clock_rows(
    session_date: date,
) -> list[tuple[int, str, int, datetime, datetime]]:
    """Return the independently specified 240 one-minute windows."""
    result: list[tuple[int, str, int, datetime, datetime]] = []
    slot = 0
    for name, start_clock in (("morning", time(9, 30)), ("afternoon", time(13, 0))):
        start = pd.Timestamp.combine(session_date, start_clock).tz_localize(TIMEZONE)
        for minute_index in range(120):
            window_start = start + pd.Timedelta(minutes=minute_index)
            window_end = window_start + pd.Timedelta(minutes=1)
            result.append((slot, name, minute_index, window_start, window_end))
            slot += 1
    return result


def expected_clock_labels_sha256_v1() -> str:
    rows = _expected_clock_rows(date(2000, 1, 1))
    labels = [pd.Timestamp(row[4]).strftime("%H:%M") for row in rows]
    return _canonical_sha256(labels)


def audit_official_minute_axis_v1(
    *,
    axis_path: Path,
    calendar_path: Path,
    source_records: list[dict[str, Any]],
    interval: tuple[str, str],
    expected_open_sessions: int,
    expected_clock_labels_sha256: str,
) -> dict[str, Any]:
    """Audit bytes, provenance, calendar coverage and exact clock semantics."""
    for path, label in ((axis_path, "axis"), (calendar_path, "calendar")):
        if path.is_symlink() or not path.is_file():
            raise CourageStrictOfficialAxisAuditError(f"unsafe or missing {label} file")
    for record in source_records:
        source = Path(str(record["path"]))
        if source.is_symlink() or not source.is_file():
            raise CourageStrictOfficialAxisAuditError(
                "unsafe or missing exchange-rule source"
            )
        if _sha256_file(source) != record["sha256"]:
            raise CourageStrictOfficialAxisAuditError("exchange-rule source byte drift")

    schema = pq.ParquetFile(axis_path).schema_arrow
    if tuple(schema.names) != EXPECTED_COLUMNS:
        raise CourageStrictOfficialAxisAuditError("official axis column order drift")
    if schema.field("session_date").type != pa.date32():
        raise CourageStrictOfficialAxisAuditError("session_date must be date32")
    for name in ("minute_window_start", "minute_window_end", "feature_ready_time"):
        field_type = schema.field(name).type
        if not pa.types.is_timestamp(field_type) or field_type.tz != TIMEZONE:
            raise CourageStrictOfficialAxisAuditError(f"{name} timezone drift")

    axis = pq.read_table(axis_path).to_pandas()
    if axis.empty or axis.isna().any().any():
        raise CourageStrictOfficialAxisAuditError(
            "official axis is empty or contains nulls"
        )
    if axis.duplicated(["session_date", "exchange", "minute_slot"]).any():
        raise CourageStrictOfficialAxisAuditError("duplicate official-minute key")
    if sorted(axis["exchange"].unique().tolist()) != list(EXPECTED_EXCHANGES):
        raise CourageStrictOfficialAxisAuditError("exchange set drift")
    if not axis["is_official_open_minute"].eq(True).all():
        raise CourageStrictOfficialAxisAuditError("closed slots must not be emitted")
    if not axis["rule_version"].eq(EXPECTED_RULE_VERSION).all():
        raise CourageStrictOfficialAxisAuditError("rule version drift")

    calendar = pq.read_table(
        calendar_path, columns=["trade_date", "is_open"]
    ).to_pandas()
    calendar["trade_date"] = pd.to_datetime(calendar["trade_date"]).dt.normalize()
    start, end = map(pd.Timestamp, interval)
    selected = calendar[
        (calendar["trade_date"] >= start) & (calendar["trade_date"] < end)
    ]
    if selected["trade_date"].duplicated().any():
        raise CourageStrictOfficialAxisAuditError("calendar date key is not unique")
    open_dates = selected.loc[selected["is_open"].eq(1), "trade_date"].tolist()
    if len(open_dates) != expected_open_sessions:
        raise CourageStrictOfficialAxisAuditError("open-session count drift")
    observed_dates = sorted(
        pd.to_datetime(axis["session_date"]).dt.normalize().unique()
    )
    if list(pd.DatetimeIndex(observed_dates)) != list(pd.DatetimeIndex(open_dates)):
        raise CourageStrictOfficialAxisAuditError(
            "axis/open-calendar date set mismatch"
        )

    if expected_clock_labels_sha256 != expected_clock_labels_sha256_v1():
        raise CourageStrictOfficialAxisAuditError(
            "independent expected-clock identity drift"
        )

    expected_rows = expected_open_sessions * len(EXPECTED_EXCHANGES) * 240
    if len(axis) != expected_rows:
        raise CourageStrictOfficialAxisAuditError("official axis row count drift")

    for (session_value, exchange), day in axis.groupby(
        ["session_date", "exchange"], sort=True, observed=True
    ):
        day = day.sort_values("minute_slot", kind="stable")
        expected = _expected_clock_rows(pd.Timestamp(session_value).date())
        if len(day) != 240 or day["minute_slot"].tolist() != list(range(240)):
            raise CourageStrictOfficialAxisAuditError(
                "session slots must be exactly 0..239"
            )
        if day["session"].tolist() != ["morning"] * 120 + ["afternoon"] * 120:
            raise CourageStrictOfficialAxisAuditError(
                "morning/afternoon segmentation drift"
            )
        if day["minute_in_session"].tolist() != list(range(120)) * 2:
            raise CourageStrictOfficialAxisAuditError("minute-in-session drift")
        for observed, wanted in zip(day.itertuples(index=False), expected, strict=True):
            slot, session_name, minute_index, window_start, window_end = wanted
            if (
                observed.minute_slot != slot
                or observed.session != session_name
                or observed.minute_in_session != minute_index
                or pd.Timestamp(observed.minute_window_start) != window_start
                or pd.Timestamp(observed.minute_window_end) != window_end
                or pd.Timestamp(observed.feature_ready_time)
                != window_end + pd.Timedelta(minutes=1)
            ):
                raise CourageStrictOfficialAxisAuditError(
                    f"clock semantic drift for {session_value}/{exchange}/{slot}"
                )

    # Both exchanges must share the same clock while preserving separate keys.
    for session_value, day in axis.groupby("session_date", sort=True, observed=True):
        left = day[day["exchange"].eq("SSE")].sort_values("minute_slot")
        right = day[day["exchange"].eq("SZSE")].sort_values("minute_slot")
        for column in (
            "minute_slot",
            "minute_window_start",
            "minute_window_end",
            "feature_ready_time",
        ):
            if (
                left[column].reset_index(drop=True).tolist()
                != right[column].reset_index(drop=True).tolist()
            ):
                raise CourageStrictOfficialAxisAuditError(
                    f"cross-exchange clock drift on {session_value}"
                )

    return {
        "decision": "PASS_C2_R1_OFFICIAL_MINUTE_AXIS_INDEPENDENT_AUDIT",
        "axis_sha256": _sha256_file(axis_path),
        "rows": len(axis),
        "open_sessions": int(expected_open_sessions),
        "exchange_sessions": int(expected_open_sessions * len(EXPECTED_EXCHANGES)),
        "exchanges": list(EXPECTED_EXCHANGES),
        "slots_per_exchange_session": 240,
        "clock_labels_sha256": expected_clock_labels_sha256,
        "timezone": TIMEZONE,
        "lunch_slots_emitted": 0,
        "feature_ready_rule": "minute_window_end_plus_one_minute",
        "exception_dates": [],
        "raw_minute_bars_used_to_infer_schedule": False,
    }

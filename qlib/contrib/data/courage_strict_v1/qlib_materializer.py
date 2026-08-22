"""Materialize Courage Strict V1 as an immutable native Qlib 1-minute bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import ipc

from qlib.contrib.data.courage_strict_v1.minute_courage_strict_c4_features_v1 import (
    DYNAMIC_FEATURES,
    SLOW_FEATURES,
)

SCHEMA_VERSION: Final[str] = "courage_strict_v1_qlib_catalog_v1"
DECISION: Final[str] = "PASS_V1_QLIB_MATERIALIZATION_AND_PARITY"
HORIZONS: Final[tuple[int, ...]] = (5, 15, 30, 60, 120, 240, 480)
AUTH_TRUE: Final[frozenset[str]] = frozenset(
    {
        "execution_authorized",
        "accepted_source_read_authorized",
        "qlib_materialization_authorized",
        "qlib_parity_authorized",
        "valid_read_authorized",
    }
)
AUTH_FALSE: Final[frozenset[str]] = frozenset(
    {
        "official_axis_extension_authorized",
        "runtime_profile_authorized",
        "feature_scaler_fit_authorized",
        "target_scaler_fit_authorized",
        "training_authorized",
        "checkpoint_selection_authorized",
        "lookback_selection_authorized",
        "final_test_provider_extension_authorized",
        "final_test_read_authorized",
        "june_read_authorized",
        "prediction_export_authorized",
        "feature_expansion_authorized",
        "output_head_expansion_authorized",
        "refit_authorized",
        "strategy_authorized",
        "backtest_authorized",
        "paper_trading_authorized",
        "live_trading_authorized",
        "order_generation_authorized",
        "remote_push_authorized",
    }
)


class CourageStrictV1QlibError(RuntimeError):
    """Raised when a V1 Qlib identity, semantic, or parity gate fails."""


@dataclass(frozen=True)
class SourceStore:
    name: str
    root: Path
    catalog: dict[str, Any]
    records: dict[tuple[str, str | None], dict[str, Any]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CourageStrictV1QlibError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CourageStrictV1QlibError(f"{label} must be a JSON object")
    return value


def _safe_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if (
        rel.is_absolute()
        or ".." in rel.parts
        or "AQuantLab" in relative
        or "ACOT" in relative
    ):
        raise CourageStrictV1QlibError(f"unsafe or legacy path: {relative}")
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()):
        raise CourageStrictV1QlibError(f"path escapes project root: {relative}")
    if path.is_symlink():
        raise CourageStrictV1QlibError(f"symlink forbidden: {relative}")
    return path


def _assert_authorization(
    authorization: dict[str, Any],
    *,
    config_path: Path,
    runner_path: Path,
) -> None:
    if authorization.get("schema_version") != "courage_strict_v1_qlib_authorization_v1":
        raise CourageStrictV1QlibError("authorization schema drift")
    if authorization.get("one_time_authorization") is not True:
        raise CourageStrictV1QlibError("authorization is not one-time")
    authorities = {
        key: value
        for key, value in authorization.items()
        if key.endswith("_authorized")
    }
    if set(authorities) != AUTH_TRUE | AUTH_FALSE:
        raise CourageStrictV1QlibError("authorization key set drift")
    if any(authorities[key] is not True for key in AUTH_TRUE):
        raise CourageStrictV1QlibError("required V1 Qlib authority missing")
    if any(authorities[key] is not False for key in AUTH_FALSE):
        raise CourageStrictV1QlibError("out-of-scope authority granted")
    for key, path in (("config", config_path), ("runner", runner_path)):
        identity = authorization.get(key)
        if not isinstance(identity, dict):
            raise CourageStrictV1QlibError(f"authorization lacks {key} identity")
        if identity.get("sha256") != sha256_file(path):
            raise CourageStrictV1QlibError(f"authorization {key} SHA drift")


def _records(catalog: dict[str, Any]) -> dict[tuple[str, str | None], dict[str, Any]]:
    result: dict[tuple[str, str | None], dict[str, Any]] = {}
    for record in catalog.get("files", []):
        if not isinstance(record, dict):
            raise CourageStrictV1QlibError("catalog file record is not an object")
        relative = str(record.get("path", ""))
        kind = str(record.get("kind", ""))
        instrument = record.get("instrument")
        if not kind:
            if relative.startswith("samples/") and relative.endswith(".parquet"):
                kind = "samples"
                instrument = Path(relative).stem
            else:
                kind = Path(relative).stem
        key = (kind, instrument)
        if key in result:
            raise CourageStrictV1QlibError(f"duplicate catalog record: {key}")
        result[key] = record
    return result


def _load_store(
    root: Path, catalog_path: Path, expected_sha: str, name: str
) -> SourceStore:
    if sha256_file(catalog_path) != expected_sha:
        raise CourageStrictV1QlibError(f"{name} catalog SHA drift")
    catalog = _read_json(catalog_path, f"{name} catalog")
    return SourceStore(name=name, root=root, catalog=catalog, records=_records(catalog))


def _record_path(store: SourceStore, kind: str, instrument: str | None = None) -> Path:
    record = store.records.get((kind, instrument))
    if record is None:
        raise CourageStrictV1QlibError(
            f"missing {store.name} record: {(kind, instrument)}"
        )
    path = _safe_path(store.root, str(record["path"]))
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise CourageStrictV1QlibError(f"{store.name} source bytes drift: {path}")
    return path


def _read_arrow(path: Path, columns: list[str] | None = None) -> pa.Table:
    with pa.memory_map(str(path), "r") as source:
        table = ipc.open_file(source).read_all()
    return table if columns is None else table.select(columns)


def _timestamp_ns(values: pd.Series | pa.Array | pa.ChunkedArray) -> np.ndarray:
    if isinstance(values, pd.Series):
        return pd.to_datetime(values, utc=True).astype("int64").to_numpy()
    return np.asarray(values.cast(pa.timestamp("ns", tz="UTC"))).view("int64")


def _symbol_to_qlib(symbol: str) -> str:
    code, exchange = symbol.split(".")
    if exchange not in {"SH", "SZ"} or len(code) != 6:
        raise CourageStrictV1QlibError(f"invalid instrument: {symbol}")
    return f"{exchange}{code}"


def _write_bin(path: Path, values: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype="<f4")
    payload = np.empty(array.size + 1, dtype="<f4")
    payload[0] = 0.0
    payload[1:] = array
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        payload.tofile(handle)
        handle.flush()
    os.replace(temporary, path)
    # Full read-after-write parity, including the Qlib start-index header.
    observed = np.fromfile(path, dtype="<f4")
    if not np.array_equal(observed.view("u4"), payload.view("u4")):
        raise CourageStrictV1QlibError(f"bin read-after-write parity failed: {path}")
    return {
        "path": path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": int(array.size),
    }


def _load_axis(store: SourceStore) -> pd.DataFrame:
    path = _record_path(store, "official_axis")
    table = _read_arrow(path)
    frame = table.to_pandas()
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    frame["feature_ready_ns"] = _timestamp_ns(frame["feature_ready_time"])
    return frame


def _load_context(store: SourceStore, start: str, end: str) -> pd.DataFrame:
    path = _record_path(store, "slow_context")
    frame = pq.read_table(path).to_pandas()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.normalize()
    return frame[(frame.signal_date >= start) & (frame.signal_date < end)].copy()


def _load_membership(store: SourceStore, start: str, end: str) -> pd.DataFrame:
    path = _record_path(store, "membership")
    frame = pq.read_table(path).to_pandas()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.normalize()
    return frame[(frame.signal_date >= start) & (frame.signal_date < end)].copy()


def _catalog_record_by_path(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(record["path"]): record for record in catalog.get("files", [])}


def _sample_path(
    store: SourceStore,
    symbol: str,
    *,
    sample_catalog_by_path: dict[str, dict[str, Any]] | None = None,
) -> Path | None:
    relative = f"samples/{symbol}.parquet"
    if sample_catalog_by_path is not None:
        record = sample_catalog_by_path.get(relative)
    else:
        record = store.records.get(("samples", symbol))
    if record is None:
        return None
    path = _safe_path(store.root, relative)
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise CourageStrictV1QlibError(f"sample bytes drift: {path}")
    return path


def _empty_values(length: int) -> dict[str, np.ndarray]:
    fields = [
        "minute_slot",
        "session_id",
        "pit_member",
        "signal_legal",
        "signal_stride5",
    ]
    fields += [
        item
        for feature in DYNAMIC_FEATURES
        for item in (feature, f"{feature}__available", f"{feature}__data_missing")
    ]
    fields += [
        item
        for feature in SLOW_FEATURES
        for item in (feature, f"{feature}__available", f"{feature}__data_missing")
    ]
    fields += ["industry_id", "industry_known"]
    fields += [
        item
        for horizon in HORIZONS
        for item in (
            f"label_return_{horizon}",
            f"label_valid_{horizon}",
            f"label_target_end_index_{horizon}",
            f"label_available_index_{horizon}",
        )
    ]
    result = {field: np.zeros(length, dtype=np.float32) for field in fields}
    for field in ("open", "high", "low", "close", "volume", "amount"):
        result[field] = np.full(length, np.nan, dtype=np.float32)
    result["bar_available"] = np.zeros(length, dtype=np.float32)
    result["bar_data_missing"] = np.ones(length, dtype=np.float32)
    return result


def _fill_axis_identity(values: dict[str, np.ndarray], axis: pd.DataFrame) -> None:
    slots = axis["minute_slot"].to_numpy(dtype=np.int16)
    values["minute_slot"][:] = slots
    values["session_id"][:] = (slots >= 120).astype(np.float32)
    legal = (slots <= 118) | ((slots >= 120) & (slots <= 238))
    legal_index = np.where(slots < 120, slots, slots - 1)
    values["signal_legal"][:] = legal.astype(np.float32)
    values["signal_stride5"][:] = (legal & (legal_index % 5 == 0)).astype(np.float32)


def _fill_raw_bars(
    values: dict[str, np.ndarray],
    *,
    path: Path,
    expected_sha256: str,
    start: str,
    end: str,
    window_start_index: dict[int, int],
) -> None:
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected_sha256:
        raise CourageStrictV1QlibError(f"minute source bytes drift: {path}")
    frame = pq.read_table(
        path,
        columns=[
            "open",
            "high",
            "low",
            "close",
            "vol",
            "amount",
            "trade_date",
            "trade_time",
        ],
    ).to_pandas()
    if "trade_date" not in frame.columns or "trade_time" not in frame.columns:
        frame = frame.reset_index()
    else:
        frame = frame.reset_index(drop=True)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame = frame[(frame.trade_date >= start) & (frame.trade_date < end)]
    local = pd.to_datetime(frame["trade_time"])
    if local.dt.tz is None:
        local = local.dt.tz_localize("Asia/Shanghai")
    else:
        local = local.dt.tz_convert("Asia/Shanghai")
    start_ns = local.dt.tz_convert("UTC").astype("int64").to_numpy()
    indices = np.fromiter(
        (window_start_index.get(int(value), -1) for value in start_ns),
        dtype=np.int64,
        count=len(frame),
    )
    selected = indices >= 0
    indices = indices[selected]
    if len(np.unique(indices)) != len(indices):
        raise CourageStrictV1QlibError(f"minute source key duplication: {path}")
    columns = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "vol",
        "amount": "amount",
    }
    raw = {
        target: pd.to_numeric(frame.loc[selected, source], errors="coerce").to_numpy(
            dtype=np.float32
        )
        for target, source in columns.items()
    }
    valid = (
        np.isfinite(raw["open"])
        & np.isfinite(raw["high"])
        & np.isfinite(raw["low"])
        & np.isfinite(raw["close"])
        & np.isfinite(raw["volume"])
        & np.isfinite(raw["amount"])
        & (raw["open"] > 0)
        & (raw["high"] >= raw["low"])
        & (raw["low"] > 0)
        & (raw["close"] > 0)
        & (raw["volume"] > 0)
        & (raw["amount"] > 0)
    )
    for field, array in raw.items():
        values[field][indices] = np.where(valid, array, np.nan)
    values["bar_available"][indices] = valid.astype(np.float32)
    values["bar_data_missing"][indices] = (~valid).astype(np.float32)


def _fill_dynamic(
    values: dict[str, np.ndarray],
    *,
    source: SourceStore,
    symbol: str,
    start: str,
    end: str,
    calendar_index: dict[int, int],
) -> None:
    record = source.records.get(("dynamic", symbol))
    if record is None:
        return
    path = _record_path(source, "dynamic", symbol)
    columns = ["session_date", "feature_ready_time"] + [
        item
        for feature in DYNAMIC_FEATURES
        for item in (feature, f"{feature}__available", f"{feature}__data_missing")
    ]
    frame = _read_arrow(path, columns).to_pandas()
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    frame = frame[(frame.session_date >= start) & (frame.session_date < end)]
    indices = np.fromiter(
        (
            calendar_index.get(int(value), -1)
            for value in _timestamp_ns(frame.feature_ready_time)
        ),
        dtype=np.int64,
        count=len(frame),
    )
    if (indices < 0).any() or len(np.unique(indices)) != len(indices):
        raise CourageStrictV1QlibError(
            f"dynamic calendar identity drift: {symbol}/{source.name}"
        )
    for feature in DYNAMIC_FEATURES:
        raw = frame[feature].to_numpy(dtype=np.float32)
        available = frame[f"{feature}__available"].to_numpy(dtype=bool)
        missing = frame[f"{feature}__data_missing"].to_numpy(dtype=bool)
        usable = available & ~missing & np.isfinite(raw)
        values[feature][indices] = np.where(usable, raw, 0.0)
        values[f"{feature}__available"][indices] = available.astype(np.float32)
        values[f"{feature}__data_missing"][indices] = missing.astype(np.float32)


def _fill_context(
    values: dict[str, np.ndarray],
    *,
    frame: pd.DataFrame,
    session_rows: dict[pd.Timestamp, np.ndarray],
) -> None:
    for row in frame.itertuples(index=False):
        date = pd.Timestamp(row.signal_date).normalize()
        indices = session_rows.get(date)
        if indices is None:
            raise CourageStrictV1QlibError(f"context date outside calendar: {date}")
        for feature in SLOW_FEATURES:
            raw = float(getattr(row, feature))
            available = bool(getattr(row, f"{feature}__available"))
            missing = bool(getattr(row, f"{feature}__data_missing"))
            usable = available and not missing and np.isfinite(raw)
            values[feature][indices] = raw if usable else 0.0
            values[f"{feature}__available"][indices] = float(available)
            values[f"{feature}__data_missing"][indices] = float(missing)


def _fill_membership(
    values: dict[str, np.ndarray],
    *,
    frame: pd.DataFrame,
    session_rows: dict[pd.Timestamp, np.ndarray],
) -> list[pd.Timestamp]:
    dates = sorted(
        pd.Timestamp(value).normalize() for value in frame.signal_date.unique()
    )
    for date in dates:
        indices = session_rows.get(date)
        if indices is None:
            raise CourageStrictV1QlibError(f"membership date outside calendar: {date}")
        values["pit_member"][indices] = 1.0
    return dates


def _fill_samples(
    values: dict[str, np.ndarray],
    *,
    sample_paths: list[tuple[Path, str, str]],
    calendar_index: dict[int, int],
    source_offset_to_index: dict[int, int],
    industry_ids: dict[str, int],
) -> int:
    rows = 0
    seen: set[int] = set()
    columns = ["signal_date", "signal_time", "industry_code", "industry_known"]
    for horizon in HORIZONS:
        columns += [
            f"gross_return_vwap1_{horizon}m",
            f"objective_gross_valid_vwap1_{horizon}m",
            f"target_global_minute_offset_vwap1_{horizon}m",
            f"label_target_available_time_vwap1_{horizon}m",
        ]
    for path, start, end in sample_paths:
        frame = pq.read_table(path, columns=columns).to_pandas()
        frame["signal_date"] = pd.to_datetime(frame.signal_date).dt.normalize()
        frame = frame[(frame.signal_date >= start) & (frame.signal_date < end)]
        signal_indices = np.fromiter(
            (
                calendar_index.get(int(value), -1)
                for value in _timestamp_ns(frame.signal_time)
            ),
            dtype=np.int64,
            count=len(frame),
        )
        if (signal_indices < 0).any():
            raise CourageStrictV1QlibError(f"sample signal outside calendar: {path}")
        for index, row in zip(
            signal_indices, frame.itertuples(index=False), strict=True
        ):
            index_int = int(index)
            if index_int in seen:
                raise CourageStrictV1QlibError(
                    f"duplicate sample signal: {path}/{index_int}"
                )
            seen.add(index_int)
            known = bool(row.industry_known)
            code = str(row.industry_code) if known else ""
            values["industry_known"][index_int] = float(known)
            values["industry_id"][index_int] = float(
                industry_ids.get(code, 0) if known else 0
            )
            for horizon in HORIZONS:
                raw = float(getattr(row, f"gross_return_vwap1_{horizon}m"))
                valid = bool(getattr(row, f"objective_gross_valid_vwap1_{horizon}m"))
                target_offset = int(
                    getattr(row, f"target_global_minute_offset_vwap1_{horizon}m")
                )
                available_ns = int(
                    pd.Timestamp(
                        getattr(row, f"label_target_available_time_vwap1_{horizon}m")
                    )
                    .tz_convert("UTC")
                    .value
                )
                target_index = source_offset_to_index.get(target_offset, -1)
                available_index = calendar_index.get(available_ns, -1)
                valid = (
                    valid
                    and np.isfinite(raw)
                    and target_index >= 0
                    and available_index >= 0
                )
                values[f"label_return_{horizon}"][index_int] = raw if valid else 0.0
                values[f"label_valid_{horizon}"][index_int] = float(valid)
                values[f"label_target_end_index_{horizon}"][index_int] = float(
                    target_index if valid else 0
                )
                values[f"label_available_index_{horizon}"][index_int] = float(
                    available_index if valid else 0
                )
        rows += len(frame)
    return rows


def _contiguous_runs(
    dates: list[pd.Timestamp],
    sessions: list[pd.Timestamp],
) -> list[tuple[int, int]]:
    positions = sorted({sessions.index(date) for date in dates})
    if not positions:
        return []
    result: list[tuple[int, int]] = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            result.append((start, previous))
            start = position
        previous = position
    result.append((start, previous))
    return result


def _build_symbol(
    symbol: str,
    *,
    staging: Path,
    axis: pd.DataFrame,
    calendar_index: dict[int, int],
    source_offset_to_index: dict[int, int],
    session_rows: dict[pd.Timestamp, np.ndarray],
    sessions: list[pd.Timestamp],
    features: SourceStore,
    label_store: SourceStore,
    label_records: dict[str, dict[str, Any]],
    contexts: dict[str, pd.DataFrame],
    memberships: dict[str, pd.DataFrame],
    industry_ids: dict[str, int],
    minute_root: Path,
    minute_records: dict[str, dict[str, Any]],
    window_start_index: dict[int, int],
) -> dict[str, Any]:
    values = _empty_values(len(axis))
    _fill_axis_identity(values, axis)
    minute_record = minute_records.get(symbol)
    if minute_record is None:
        raise CourageStrictV1QlibError(f"minute source coverage missing: {symbol}")
    _fill_raw_bars(
        values,
        path=_safe_path(minute_root, str(minute_record["relative_path"])),
        expected_sha256=str(minute_record["sha256"]),
        start="2025-04-01",
        end="2026-04-01",
        window_start_index=window_start_index,
    )
    _fill_dynamic(
        values,
        source=features,
        symbol=symbol,
        start="2025-04-01",
        end="2026-04-01",
        calendar_index=calendar_index,
    )
    context = contexts.get(symbol, pd.DataFrame())
    if not context.empty:
        _fill_context(values, frame=context, session_rows=session_rows)
    membership = memberships.get(symbol, pd.DataFrame())
    member_dates = _fill_membership(values, frame=membership, session_rows=session_rows)
    sample_paths: list[tuple[Path, str, str]] = []
    old_sample = _sample_path(label_store, symbol, sample_catalog_by_path=label_records)
    if old_sample is not None:
        sample_paths.append((old_sample, "2025-07-01", "2026-04-01"))
    sample_rows = _fill_samples(
        values,
        sample_paths=sample_paths,
        calendar_index=calendar_index,
        source_offset_to_index=source_offset_to_index,
        industry_ids=industry_ids,
    )
    directory = staging / "features" / _symbol_to_qlib(symbol).lower()
    records: list[dict[str, Any]] = []
    for field in sorted(values):
        record = _write_bin(directory / f"{field.lower()}.1min.bin", values[field])
        record.update({"instrument": symbol, "field": field, "kind": "feature_bin"})
        record["path"] = record["path"].relative_to(staging).as_posix()
        records.append(record)
    return {
        "symbol": symbol,
        "qlib_instrument": _symbol_to_qlib(symbol),
        "records": records,
        "sample_rows": sample_rows,
        "member_dates": member_dates,
        "target_runs": _contiguous_runs(member_dates, sessions),
    }


def _verify_existing(root: Path, expected: dict[str, Any]) -> dict[str, Any]:
    manifest_path = root / "_courage_strict_v1_qlib_catalog.json"
    if not manifest_path.is_file():
        raise CourageStrictV1QlibError("existing provider lacks terminal catalog")
    manifest = _read_json(manifest_path, "existing V1 Qlib catalog")
    for key in ("schema_version", "decision", "config_sha256", "authorization_sha256"):
        if manifest.get(key) != expected.get(key):
            raise CourageStrictV1QlibError(f"existing V1 identity drift: {key}")
    records = manifest.get("files", [])
    actual_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "_courage_strict_v1_qlib_catalog.json"
    )
    expected_paths = sorted(str(record["path"]) for record in records)
    if actual_paths != expected_paths:
        raise CourageStrictV1QlibError("existing V1 exact tree drift")
    for record in records:
        path = _safe_path(root, str(record["path"]))
        if (
            path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise CourageStrictV1QlibError(f"existing V1 file drift: {path}")
    if canonical_sha256(records) != manifest.get("file_set_sha256"):
        raise CourageStrictV1QlibError("existing V1 file-set root drift")
    return manifest


def run_courage_strict_v1_qlib(
    *,
    project_root: Path,
    config_path: Path,
    authorization_path: Path,
) -> dict[str, Any]:
    root = project_root.resolve()
    config_path = config_path.resolve()
    authorization_path = authorization_path.resolve()
    config = _read_json(config_path, "V1 Qlib config")
    authorization = _read_json(authorization_path, "V1 Qlib authorization")
    runner_path = Path(__file__).resolve()
    _assert_authorization(
        authorization,
        config_path=config_path,
        runner_path=runner_path,
    )
    if config.get("schema_version") != "courage_strict_v1_qlib_config_v1":
        raise CourageStrictV1QlibError("config schema drift")
    output_root = _safe_path(root, config["output_root"])
    expected = {
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION,
        "config_sha256": sha256_file(config_path),
        "authorization_sha256": sha256_file(authorization_path),
    }
    if output_root.exists():
        return _verify_existing(output_root, expected)

    stores: dict[str, SourceStore] = {}
    for name, item in config["stores"].items():
        catalog_path = _safe_path(root, item["catalog_path"])
        stores[name] = _load_store(
            _safe_path(root, item["root"]),
            catalog_path,
            item["catalog_sha256"],
            name,
        )
    features, label_store = stores["features"], stores["labels"]

    minute_item = config["minute_source"]
    minute_root = _safe_path(root, minute_item["root"])
    minute_identity_path = _safe_path(root, minute_item["identity_path"])
    if sha256_file(minute_identity_path) != minute_item["identity_sha256"]:
        raise CourageStrictV1QlibError("minute source identity SHA drift")
    minute_identity = _read_json(minute_identity_path, "minute source identity")
    if minute_identity.get(
        "decision"
    ) != "PASS_V1_MINUTE_SOURCE_CONSTRUCTED" or minute_identity.get("interval") != [
        "2025-04-01",
        "2026-05-01",
    ]:
        raise CourageStrictV1QlibError("minute source identity drift")
    minute_records = {
        Path(str(item["relative_path"])).stem: item
        for item in minute_identity.get("files", [])
    }

    axis = _load_axis(features)
    axis = axis[
        (axis.session_date >= "2025-04-01") & (axis.session_date < "2026-04-01")
    ].copy()
    axis = axis.sort_values("feature_ready_ns", kind="mergesort").reset_index(drop=True)
    if len(axis) != 58_080 or axis.session_date.nunique() != 242:
        raise CourageStrictV1QlibError("V1 calendar count drift")
    calendar_ns = axis.feature_ready_ns.to_numpy(dtype=np.int64)
    if len(np.unique(calendar_ns)) != len(calendar_ns) or np.any(
        np.diff(calendar_ns) <= 0
    ):
        raise CourageStrictV1QlibError("calendar is not strictly increasing and unique")
    calendar_index = {int(value): index for index, value in enumerate(calendar_ns)}
    window_start_ns = _timestamp_ns(axis.minute_window_start)
    window_start_index = {
        int(value): index for index, value in enumerate(window_start_ns)
    }
    source_offset_to_index = {
        int(offset): calendar_index[int(timestamp)]
        for offset, timestamp in zip(
            axis.global_minute_offset, axis.feature_ready_ns, strict=True
        )
    }
    session_rows = {
        pd.Timestamp(date).normalize(): group.index.to_numpy(dtype=np.int64)
        for date, group in axis.groupby("session_date", sort=True)
    }
    if any(len(indices) != 240 for indices in session_rows.values()):
        raise CourageStrictV1QlibError("calendar session does not have 240 slots")
    sessions = sorted(session_rows)

    contexts = _load_context(features, "2025-07-01", "2026-04-01")
    memberships = contexts.loc[:, ["instrument", "signal_date"]].copy()
    if memberships.duplicated(["instrument", "signal_date"]).any():
        raise CourageStrictV1QlibError("membership duplicate key")
    symbols = sorted(memberships.instrument.unique())
    if set(symbols) - set(minute_records):
        raise CourageStrictV1QlibError("minute source lacks a selected instrument")
    context_groups = {
        symbol: frame.copy() for symbol, frame in contexts.groupby("instrument")
    }
    membership_groups = {
        symbol: frame.copy() for symbol, frame in memberships.groupby("instrument")
    }

    # Vocabulary is derived only from Train/Valid samples.
    industry_codes: set[str] = set()
    catalog_by_path = _catalog_record_by_path(label_store.catalog)
    for symbol in symbols:
        path = _sample_path(
            label_store,
            symbol,
            sample_catalog_by_path=catalog_by_path,
        )
        if path is None:
            continue
        frame = pq.read_table(
            path,
            columns=["signal_date", "industry_code", "industry_known"],
        ).to_pandas()
        frame["signal_date"] = pd.to_datetime(frame.signal_date).dt.normalize()
        frame = frame[
            (frame.signal_date >= "2025-07-01")
            & (frame.signal_date < "2026-04-01")
            & frame.industry_known
        ]
        industry_codes.update(
            str(value) for value in frame.industry_code.dropna().unique()
        )
    industry_ids = {
        code: index + 1 for index, code in enumerate(sorted(industry_codes))
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    scratch_root = output_root.parent / ".scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    staging = (
        scratch_root / f"{output_root.name}-{expected['authorization_sha256'][:20]}"
    )
    for stale in scratch_root.glob(f"{output_root.name}-*"):
        if stale == staging:
            continue
        if stale.is_symlink() or not stale.is_dir():
            raise CourageStrictV1QlibError(f"unsafe stale staging entry: {stale}")
        shutil.rmtree(stale)
    if staging.exists():
        if staging.is_symlink():
            raise CourageStrictV1QlibError("V1 Qlib staging root is a symlink")
        shutil.rmtree(staging)
    staging.mkdir()
    (staging / "calendars").mkdir()
    (staging / "instruments").mkdir()
    records: list[dict[str, Any]] = []
    try:
        calendar_path = staging / "calendars/1min.txt"
        calendar_text = (
            "\n".join(
                pd.Timestamp(value)
                .tz_convert("Asia/Shanghai")
                .strftime("%Y-%m-%d %H:%M:%S")
                for value in pd.to_datetime(calendar_ns, utc=True)
            )
            + "\n"
        )
        calendar_path.write_text(calendar_text, encoding="utf-8")
        records.append(
            {
                "kind": "calendar",
                "path": "calendars/1min.txt",
                "rows": len(axis),
                "bytes": calendar_path.stat().st_size,
                "sha256": sha256_file(calendar_path),
            }
        )
        label_records = _catalog_record_by_path(label_store.catalog)
        results: list[dict[str, Any]] = []
        workers = int(config["workers"])
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _build_symbol,
                    symbol,
                    staging=staging,
                    axis=axis,
                    calendar_index=calendar_index,
                    source_offset_to_index=source_offset_to_index,
                    session_rows=session_rows,
                    sessions=sessions,
                    features=features,
                    label_store=label_store,
                    label_records=label_records,
                    contexts=context_groups,
                    memberships=membership_groups,
                    industry_ids=industry_ids,
                    minute_root=minute_root,
                    minute_records=minute_records,
                    window_start_index=window_start_index,
                ): symbol
                for symbol in symbols
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if completed % 100 == 0 or completed == len(futures):
                    print(
                        f"Qlib provider instruments {completed}/{len(futures)}",
                        flush=True,
                    )
        results.sort(key=lambda item: item["symbol"])
        all_lines: list[str] = []
        target_lines: list[str] = []
        for result in results:
            records.extend(result["records"])
            first_time = pd.Timestamp(axis.iloc[0].feature_ready_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            last_time = pd.Timestamp(axis.iloc[-1].feature_ready_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            all_lines.append(f"{result['qlib_instrument']}\t{first_time}\t{last_time}")
            for start_index, end_index in result["target_runs"]:
                start_time = pd.Timestamp(
                    axis.iloc[session_rows[sessions[start_index]][0]].feature_ready_time
                ).strftime("%Y-%m-%d %H:%M:%S")
                end_time = pd.Timestamp(
                    axis.iloc[session_rows[sessions[end_index]][-1]].feature_ready_time
                ).strftime("%Y-%m-%d %H:%M:%S")
                target_lines.append(
                    f"{result['qlib_instrument']}\t{start_time}\t{end_time}"
                )
        for relative, lines, kind in (
            ("instruments/all.txt", all_lines, "instruments_all"),
            ("instruments/turnover_5_15_pit.txt", target_lines, "instruments_target"),
        ):
            path = staging / relative
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            records.append(
                {
                    "kind": kind,
                    "path": relative,
                    "rows": len(lines),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        vocabulary_path = staging / "industry_vocabulary.json"
        vocabulary_path.write_text(
            json.dumps(
                {"UNKNOWN": 0, **industry_ids},
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        records.append(
            {
                "kind": "industry_vocabulary",
                "path": "industry_vocabulary.json",
                "rows": len(industry_ids) + 1,
                "bytes": vocabulary_path.stat().st_size,
                "sha256": sha256_file(vocabulary_path),
            }
        )
        records.sort(key=lambda item: str(item["path"]))
        manifest = {
            **expected,
            "run_id": "courage_strict_v1",
            "provider_scope": ["2025-04-01", "2026-04-01"],
            "train_scope": ["2025-07-01", "2026-03-02"],
            "valid_scope": ["2026-03-02", "2026-04-01"],
            "june_read": False,
            "may_read": False,
            "source": "courage_strict_v1_project_owned_feature_and_label_stores",
            "coverage": {
                "calendar_sessions": len(sessions),
                "calendar_rows": len(axis),
                "instruments": len(symbols),
                "membership_rows": len(memberships),
                "sample_rows": sum(result["sample_rows"] for result in results),
                "fields_per_instrument": len(results[0]["records"]),
                "bin_files": sum(len(result["records"]) for result in results),
                "raw_bar_fields": 8,
                "industry_vocabulary": len(industry_ids) + 1,
            },
            "parity": {
                "source_catalog_SHA_verified_before_read": True,
                "source_file_SHA_verified_before_read": True,
                "calendar_overlap_exact": True,
                "float32_source_cast_then_bit_exact_bin_readback": True,
                "explicit_masks_preserved": True,
                "raw_OHLCVA_and_bar_masks_materialized": True,
                "labels_and_timing_indices_preserved": True,
                "full_file_read_after_write": True,
            },
            "file_set_sha256": canonical_sha256(records),
            "files": records,
            "authorization_consumed": True,
            "terminal_authority_matrix": {
                key: False for key in sorted(AUTH_TRUE | AUTH_FALSE)
            },
            "runtime_profile_executed": False,
            "training_executed": False,
            "checkpoint_or_lookback_selection_executed": False,
            "final_test_read": False,
            "refit_strategy_backtest_trading_remote_push_executed": False,
        }
        manifest_path = staging / "_courage_strict_v1_qlib_catalog.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        # One durability barrier after all per-file bit-exact readbacks avoids
        # hundreds of thousands of tiny fsync operations.
        os.sync()
        verified = _verify_existing(staging, expected)
        os.replace(staging, output_root)
        # Atomic rename preserves the already verified tree and bytes. A later
        # invocation still performs the full immutable-tree verification.
        return verified
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_courage_strict_v1_qlib(
        project_root=args.project_root,
        config_path=args.config,
        authorization_path=args.authorization,
    )
    print(
        json.dumps(
            {key: result[key] for key in ("decision", "coverage", "file_set_sha256")},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

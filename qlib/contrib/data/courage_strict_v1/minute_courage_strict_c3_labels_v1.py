"""C3-B time-safe VWAP1 labels and immutable sample catalog.

This runner consumes only the accepted main-board C3-A population and minute
records strictly before the frozen role cutoff.  It does not build features,
read Development Test, or train a model.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from qlib.contrib.data.courage_strict_v1.minute_courage_strict_c2_identity_v1 import (
    sha256_file_v1,
)


class CourageStrictC3LabelError(RuntimeError):
    """Raised when the C3-B label/sample identity fails closed."""


CONFIG_SCHEMA: Final[str] = "courage_strict_c3_label_config_v1"
AUTH_SCHEMA: Final[str] = "courage_strict_c3_label_authorization_v1"
MANIFEST_SCHEMA: Final[str] = "courage_strict_c3_label_manifest_v1"
HORIZONS: Final[tuple[int, ...]] = (5, 15, 30, 60, 120, 240, 480)
OPERATOR_STATEMENT: Final[str] = (
    "用户授权在当前Qlib仓库完成courage_strict_v1全部代码、数据构造、训练和评测。"
)
TRUE_AUTHORITIES: Final[set[str]] = {
    "execution_authorized",
    "accepted_C2_C3A_read_authorized",
    "minute_bar_train_valid_record_read_authorized",
    "label_build_authorized",
    "sample_catalog_materialization_authorized",
    "C3_evidence_write_authorized",
}
FALSE_AUTHORITIES: Final[set[str]] = {
    "feature_build_authorized",
    "sequence_build_authorized",
    "runtime_profile_authorized",
    "training_authorized",
    "model_selection_authorized",
    "development_test_read_authorized",
    "reserved_confirm_read_authorized",
    "holdout_read_authorized",
    "refit_authorized",
    "strategy_authorized",
    "backtest_authorized",
    "paper_trading_authorized",
    "live_trading_authorized",
    "order_generation_authorized",
    "remote_push_authorized",
}

STATUS_VALID: Final[str] = "VALID"
STATUS_ROLE_CUTOFF: Final[str] = "ROLE_CUTOFF_OR_AXIS_END"
STATUS_ENTRY_MISSING: Final[str] = "ENTRY_MISSING_OR_INVALID"
STATUS_TARGET_MISSING: Final[str] = "TARGET_MISSING_OR_INVALID"
STATUS_ENTRY_LIMIT_LOCKED: Final[str] = "ENTRY_LIMIT_LOCKED"
STATUS_TARGET_LIMIT_LOCKED: Final[str] = "TARGET_LIMIT_LOCKED"
STATUS_ACTION_CROSSING: Final[str] = "CORPORATE_ACTION_CROSSING"


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CourageStrictC3LabelError(f"unsafe or missing {label}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CourageStrictC3LabelError(f"{label} must be an object")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if path.is_symlink() or not path.is_file():
        raise CourageStrictC3LabelError(f"unsafe or missing input: {value}")
    return path


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()


def _canonical_root(records: list[dict[str, Any]]) -> str:
    identities = [
        {key: item[key] for key in ("path", "sha256", "bytes", "rows")}
        for item in sorted(records, key=lambda value: value["path"])
    ]
    return hashlib.sha256(_canonical_json(identities)).hexdigest()


def _minute_record_map(identity: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in identity.get("files", []):
        if item.get("roles") != ["one_minute_OHLCVA_bars"]:
            continue
        instrument = Path(item["relative_path"]).stem
        if instrument in result:
            raise CourageStrictC3LabelError("duplicate source minute identity")
        result[instrument] = item
    return result


def _validate_axis(axis: pd.DataFrame) -> pd.DataFrame:
    required = {
        "session_date",
        "exchange",
        "minute_slot",
        "minute_window_start",
        "minute_window_end",
        "feature_ready_time",
        "is_official_open_minute",
    }
    if not required.issubset(axis.columns):
        raise CourageStrictC3LabelError("official axis schema drift")
    frame = axis.loc[axis["exchange"].eq("SSE")].copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    for column in ("minute_window_start", "minute_window_end", "feature_ready_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True).dt.tz_convert(
            "Asia/Shanghai"
        )
    frame = frame.sort_values(
        ["session_date", "minute_slot"], kind="mergesort"
    ).reset_index(drop=True)
    if frame.empty or not frame["is_official_open_minute"].eq(True).all():
        raise CourageStrictC3LabelError("official axis open-minute drift")
    if frame.groupby("session_date").size().ne(240).any():
        raise CourageStrictC3LabelError("official axis is not 240 slots/day")
    expected_slots = np.tile(
        np.arange(240, dtype=np.int16), frame["session_date"].nunique()
    )
    if not np.array_equal(
        frame["minute_slot"].to_numpy(dtype=np.int16), expected_slots
    ):
        raise CourageStrictC3LabelError("official axis slot order drift")
    start = frame["minute_window_start"].astype("int64").to_numpy()
    end = frame["minute_window_end"].astype("int64").to_numpy()
    ready = frame["feature_ready_time"].astype("int64").to_numpy()
    minute_ns = 60 * 1_000_000_000
    if not np.all(end - start == minute_ns) or not np.all(ready - end == minute_ns):
        raise CourageStrictC3LabelError("official axis readiness drift")
    frame["global_minute_offset"] = np.arange(len(frame), dtype=np.int64)
    return frame


def _bar_is_valid(
    volume: np.ndarray, amount: np.ndarray, price: np.ndarray
) -> np.ndarray:
    return (
        np.isfinite(volume)
        & np.isfinite(amount)
        & np.isfinite(price)
        & (volume > 0)
        & (amount > 0)
        & (price > 0)
    )


def _limit_locked(
    *,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
) -> np.ndarray:
    one_price = (
        np.isfinite(open_)
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(close)
        & np.isclose(open_, high, rtol=0.0, atol=1e-10)
        & np.isclose(high, low, rtol=0.0, atol=1e-10)
        & np.isclose(low, close, rtol=0.0, atol=1e-10)
    )
    at_upper = np.isfinite(upper) & np.isclose(close, upper, rtol=0.0, atol=0.005)
    at_lower = np.isfinite(lower) & np.isclose(close, lower, rtol=0.0, atol=0.005)
    return one_price & (at_upper | at_lower)


def build_symbol_sample_partition_v1(
    *,
    instrument: str,
    membership: pd.DataFrame,
    grid: pd.DataFrame,
    axis: pd.DataFrame,
    bars: pd.DataFrame,
    industry: pd.DataFrame,
    daily_status: pd.DataFrame,
    action_dates: np.ndarray,
    role_cutoffs: dict[str, pd.Timestamp],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one deterministic wide sample partition for a single instrument."""
    if membership.empty:
        raise CourageStrictC3LabelError("empty symbol membership")
    expected = len(membership) * 48
    sample = membership.merge(
        grid,
        left_on="signal_date",
        right_on="session_date",
        how="left",
        validate="many_to_many",
    )
    if len(sample) != expected or sample["signal_time"].isna().any():
        raise CourageStrictC3LabelError("symbol sample/grid cardinality drift")
    sample = sample.sort_values("signal_time", kind="mergesort").reset_index(drop=True)
    sample["instrument"] = instrument
    sample["sample_id"] = (
        sample["instrument"].astype(str)
        + "|"
        + sample["signal_time"].astype("int64").astype(str)
    )
    if sample["sample_id"].duplicated().any():
        raise CourageStrictC3LabelError("duplicate sample id")

    axis_size = len(axis)
    date_by_offset = axis["session_date"].to_numpy(dtype="datetime64[ns]")
    start_by_offset = axis["minute_window_start"].to_numpy()
    end_by_offset = axis["minute_window_end"].to_numpy()
    ready_by_offset = axis["feature_ready_time"].to_numpy()

    vwap = np.full(axis_size, np.nan, dtype=np.float64)
    open_ = np.full(axis_size, np.nan, dtype=np.float64)
    high = np.full(axis_size, np.nan, dtype=np.float64)
    low = np.full(axis_size, np.nan, dtype=np.float64)
    close = np.full(axis_size, np.nan, dtype=np.float64)
    if not bars.empty:
        raw = bars.copy()
        raw["trade_time"] = pd.to_datetime(raw["trade_time"], errors="coerce")
        if raw["trade_time"].isna().any() or raw["trade_time"].duplicated().any():
            raise CourageStrictC3LabelError("minute timestamp is invalid or duplicate")
        end_lookup = pd.Series(
            axis["global_minute_offset"].to_numpy(),
            index=axis["minute_window_end"].dt.tz_localize(None).astype("int64"),
        )
        offsets = raw["trade_time"].astype("int64").map(end_lookup)
        raw = raw.loc[offsets.notna()].copy()
        offsets = offsets.loc[offsets.notna()].to_numpy(dtype=np.int64)
        volume = pd.to_numeric(raw["vol"], errors="coerce").to_numpy(dtype=np.float64)
        amount = pd.to_numeric(raw["amount"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        price = np.divide(
            amount,
            volume,
            out=np.full(len(raw), np.nan, dtype=np.float64),
            where=volume > 0,
        )
        valid = _bar_is_valid(volume, amount, price)
        vwap[offsets[valid]] = price[valid]
        for target, column in (
            (open_, "open"),
            (high, "high"),
            (low, "low"),
            (close, "close"),
        ):
            target[offsets] = pd.to_numeric(raw[column], errors="coerce").to_numpy(
                dtype=np.float64
            )

    status = daily_status.copy()
    status["session_date"] = pd.to_datetime(status["session_date"]).dt.normalize()
    status = status.set_index("session_date")
    axis_dates = pd.DatetimeIndex(axis["session_date"])
    upper = status["limit_up_price"].reindex(axis_dates).to_numpy(dtype=np.float64)
    lower = status["limit_down_price"].reindex(axis_dates).to_numpy(dtype=np.float64)
    suspended = (
        status["is_suspended"]
        .reindex(axis_dates)
        .astype("boolean")
        .fillna(True)
        .to_numpy(dtype=bool)
    )
    locked = _limit_locked(
        open_=open_, high=high, low=low, close=close, upper=upper, lower=lower
    )
    usable_bar = np.isfinite(vwap) & ~suspended & ~locked

    industry_frame = industry.copy()
    industry_frame["session_date"] = pd.to_datetime(
        industry_frame["session_date"]
    ).dt.normalize()
    sample = sample.merge(
        industry_frame.loc[
            :,
            [
                "session_date",
                "industry_code",
                "sector_level2_code",
                "industry_known",
                "change_date",
            ],
        ],
        left_on="signal_date",
        right_on="session_date",
        how="left",
        validate="many_to_one",
        suffixes=("", "_industry"),
    )
    # G10 is keyed to observed daily rows.  A selected member can have no
    # same-day daily row (for example, a current-day suspension).  Retain the
    # sample but fail its slow categorical context closed to UNKNOWN; never
    # backfill it from a future observation.
    missing_industry = sample["industry_known"].isna()
    sample.loc[missing_industry, "industry_code"] = None
    sample.loc[missing_industry, "sector_level2_code"] = None
    sample.loc[missing_industry, "change_date"] = pd.NaT
    sample["industry_known"] = sample["industry_known"].astype("boolean").fillna(False)
    entry = sample["entry_global_minute_offset"].to_numpy(dtype=np.int64)
    if (entry < 0).any() or (entry >= axis_size).any():
        raise CourageStrictC3LabelError("entry offset outside official axis")
    sample["entry_window_start"] = start_by_offset[entry]
    sample["entry_window_end"] = end_by_offset[entry]
    sample["entry_available_time"] = ready_by_offset[entry]
    sample["entry_vwap1"] = vwap[entry]
    entry_valid = usable_bar[entry]
    entry_locked = locked[entry]
    entry_dates = date_by_offset[entry]
    action_dates = np.sort(np.asarray(action_dates, dtype="datetime64[ns]"))
    valid_counts: dict[str, int] = {}
    status_counts: dict[str, dict[str, int]] = {}

    for horizon in HORIZONS:
        target = entry + horizon
        in_axis = target < axis_size
        safe_target = np.minimum(target, axis_size - 1)
        target_dates = date_by_offset[safe_target]
        cutoff_ns = np.array(
            [role_cutoffs[str(role)].value for role in sample["role"]], dtype=np.int64
        )
        target_end_ns = pd.DatetimeIndex(end_by_offset[safe_target]).astype("int64")
        target_ready_ns = pd.DatetimeIndex(ready_by_offset[safe_target]).astype("int64")
        mature = in_axis & (target_end_ns < cutoff_ns) & (target_ready_ns < cutoff_ns)
        target_valid = mature & usable_bar[safe_target]
        crosses = np.zeros(len(sample), dtype=bool)
        if len(action_dates):
            left = np.searchsorted(action_dates, entry_dates, side="right")
            right = np.searchsorted(action_dates, target_dates, side="right")
            crosses = mature & (right > left)
        valid = entry_valid & target_valid & ~crosses
        raw_return = np.full(len(sample), np.nan, dtype=np.float64)
        raw_return[valid] = vwap[safe_target[valid]] / vwap[entry[valid]] - 1.0
        valid &= np.isfinite(raw_return)
        raw_return[~valid] = np.nan
        head_status = np.full(len(sample), STATUS_VALID, dtype=object)
        head_status[~mature] = STATUS_ROLE_CUTOFF
        head_status[mature & ~entry_valid] = STATUS_ENTRY_MISSING
        head_status[mature & entry_locked] = STATUS_ENTRY_LIMIT_LOCKED
        head_status[mature & entry_valid & ~usable_bar[safe_target]] = (
            STATUS_TARGET_MISSING
        )
        head_status[mature & entry_valid & locked[safe_target]] = (
            STATUS_TARGET_LIMIT_LOCKED
        )
        head_status[mature & entry_valid & usable_bar[safe_target] & crosses] = (
            STATUS_ACTION_CROSSING
        )
        sample[f"gross_return_vwap1_{horizon}m"] = raw_return
        sample[f"objective_gross_valid_vwap1_{horizon}m"] = valid
        sample[f"target_global_minute_offset_vwap1_{horizon}m"] = np.where(
            in_axis, target, -1
        ).astype(np.int64)
        sample[f"target_window_end_vwap1_{horizon}m"] = pd.to_datetime(
            np.where(in_axis, target_end_ns, np.datetime64("NaT").astype("int64")),
            utc=True,
        ).tz_convert("Asia/Shanghai")
        sample[f"label_target_available_time_vwap1_{horizon}m"] = pd.to_datetime(
            np.where(in_axis, target_ready_ns, np.datetime64("NaT").astype("int64")),
            utc=True,
        ).tz_convert("Asia/Shanghai")
        sample[f"label_status_vwap1_{horizon}m"] = head_status
        valid_counts[str(horizon)] = int(valid.sum())
        values, counts = np.unique(head_status, return_counts=True)
        status_counts[str(horizon)] = {
            str(value): int(count) for value, count in zip(values, counts, strict=True)
        }

    drop_columns = [
        column
        for column in (
            "session_date",
            "session_date_industry",
            "global_minute_offset",
        )
        if column in sample
    ]
    sample = sample.drop(columns=drop_columns)
    if len(sample) != expected:
        raise CourageStrictC3LabelError("sample cardinality changed")
    for horizon in HORIZONS:
        mask = sample[f"objective_gross_valid_vwap1_{horizon}m"].to_numpy(dtype=bool)
        values = sample[f"gross_return_vwap1_{horizon}m"].to_numpy(dtype=np.float64)
        if not np.isfinite(values[mask]).all() or np.isfinite(values[~mask]).any():
            raise CourageStrictC3LabelError("value/mask invariant failed")
    return sample, {
        "rows": len(sample),
        "valid_counts": valid_counts,
        "status_counts": status_counts,
        "unknown_industry_rows": int(missing_industry.sum()),
    }


def _validate_controls(
    root: Path, config_path: Path, auth_path: Path
) -> dict[str, Any]:
    config = _load_json(config_path, "C3-B config")
    auth = _load_json(auth_path, "C3-B authorization")
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("status") != (
        "FROZEN_NOT_AUTHORIZED"
    ):
        raise CourageStrictC3LabelError("config identity drift")
    if any(value is not False for value in config["authority_matrix"].values()):
        raise CourageStrictC3LabelError("config authority drift")
    for label, identity in config["inputs"].items():
        if sha256_file_v1(_resolve(root, identity["path"])) != identity["sha256"]:
            raise CourageStrictC3LabelError(f"input drift: {label}")
    if auth.get("schema_version") != AUTH_SCHEMA or auth.get("operator_statement") != (
        OPERATOR_STATEMENT
    ):
        raise CourageStrictC3LabelError("authorization identity drift")
    if (
        auth.get("operator_statement_sha256")
        != hashlib.sha256(OPERATOR_STATEMENT.encode()).hexdigest()
    ):
        raise CourageStrictC3LabelError("operator statement SHA drift")
    if (
        auth.get("one_time_authorization") is not True
        or auth.get("consumed_when_manifest_written") is not True
    ):
        raise CourageStrictC3LabelError("authorization lifecycle drift")
    if auth.get("config") != {
        "path": config_path.relative_to(root).as_posix(),
        "sha256": sha256_file_v1(config_path),
    }:
        raise CourageStrictC3LabelError("authorization config drift")
    runner = Path(__file__).resolve()
    if auth.get("runner") != {
        "path": runner.relative_to(root).as_posix(),
        "sha256": sha256_file_v1(runner),
    }:
        raise CourageStrictC3LabelError("authorization runner drift")
    authorities = auth.get("authorities")
    if not isinstance(authorities, dict) or set(authorities) != (
        TRUE_AUTHORITIES | FALSE_AUTHORITIES
    ):
        raise CourageStrictC3LabelError("authorization key drift")
    if any(authorities[key] is not True for key in TRUE_AUTHORITIES) or any(
        authorities[key] is not False for key in FALSE_AUTHORITIES
    ):
        raise CourageStrictC3LabelError("authorization scope drift")
    return config


def _validate_existing(target: Path, *, config_sha: str, auth_sha: str) -> Path | None:
    manifest_path = target / "_c3_label_manifest.json"
    if not target.exists():
        return None
    manifest = _load_json(manifest_path, "existing C3-B manifest")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("decision") != "PASS_C3_B_LABEL_AND_SAMPLE_CATALOG"
        or manifest.get("config_sha256") != config_sha
        or manifest.get("authorization_sha256") != auth_sha
        or manifest.get("development_test_read") is not False
        or manifest.get("training_executed") is not False
    ):
        raise CourageStrictC3LabelError("existing terminal manifest drift")
    records = manifest.get("files")
    if (
        not isinstance(records, list)
        or len(records) != manifest["coverage"]["sample_partitions"]
    ):
        raise CourageStrictC3LabelError("existing file inventory drift")
    expected = {"_c3_label_manifest.json"} | {item["path"] for item in records}
    actual = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise CourageStrictC3LabelError("existing exact tree drift")
    for item in records:
        path = target / item["path"]
        if path.is_symlink() or sha256_file_v1(path) != item["sha256"]:
            raise CourageStrictC3LabelError("existing partition SHA drift")
        if pq.ParquetFile(path).metadata.num_rows != item["rows"]:
            raise CourageStrictC3LabelError("existing partition row drift")
    if _canonical_root(records) != manifest["partition_set_sha256"]:
        raise CourageStrictC3LabelError("existing partition root drift")
    return manifest_path


def run_courage_strict_c3_labels_v1(
    *, project_root: Path, config_path: Path, authorization_path: Path
) -> Path:
    root = project_root.resolve()
    config_path = config_path.resolve()
    authorization_path = authorization_path.resolve()
    config = _validate_controls(root, config_path, authorization_path)
    config_sha = sha256_file_v1(config_path)
    auth_sha = sha256_file_v1(authorization_path)
    target = root / config["output_root"] / f"authorization-{auth_sha[:24]}"
    existing = _validate_existing(target, config_sha=config_sha, auth_sha=auth_sha)
    if existing is not None:
        return existing

    inputs = {
        label: _resolve(root, value["path"])
        for label, value in config["inputs"].items()
    }
    membership_manifest = _load_json(inputs["c3_membership_manifest"], "C3-A manifest")
    if membership_manifest.get("decision") != "PASS_C3_A_MEMBERSHIP_AND_SIGNAL_GRID":
        raise CourageStrictC3LabelError("C3-A is not accepted")
    membership = pd.read_parquet(inputs["membership"])
    grid = pd.read_parquet(inputs["signal_grid"])
    if len(membership) != config["expected_counts"]["membership_rows"]:
        raise CourageStrictC3LabelError("membership row drift")
    if int(len(membership) * 48) != config["expected_counts"]["samples"]:
        raise CourageStrictC3LabelError("expected sample count drift")
    if set(membership["role"]) != {"train", "valid"}:
        raise CourageStrictC3LabelError("role drift")
    if not membership["instrument"].astype(str).str.endswith((".SH", ".SZ")).all():
        raise CourageStrictC3LabelError("non-mainboard instrument leaked into C3-B")

    axis = _validate_axis(pd.read_parquet(inputs["official_axis"]))
    industry = pd.read_parquet(inputs["daily_industry"])
    status = pd.read_parquet(inputs["normalized_status"])
    actions = pd.read_parquet(inputs["accepted_actions"])
    quarantine = pd.read_parquet(inputs["quarantined_actions"])
    identity = _load_json(inputs["source_identity_manifest"], "source identity")
    minute_records = _minute_record_map(identity)
    symbols = sorted(membership["instrument"].astype(str).unique())
    if set(symbols) - set(minute_records):
        raise CourageStrictC3LabelError("membership minute source missing")

    membership_by_symbol = {
        key: frame.copy() for key, frame in membership.groupby("instrument", sort=True)
    }
    industry = industry.loc[industry["instrument"].isin(symbols)].copy()
    industry_by_symbol = {
        key: frame.copy() for key, frame in industry.groupby("instrument", sort=True)
    }
    status = status.loc[status["instrument"].isin(symbols)].copy()
    status_by_symbol = {
        key: frame.copy() for key, frame in status.groupby("instrument", sort=True)
    }
    action_by_symbol: dict[str, np.ndarray] = {}
    for symbol in symbols:
        accepted_dates = pd.to_datetime(
            actions.loc[actions["instrument"].eq(symbol), "ex_date"]
        ).to_numpy(dtype="datetime64[ns]")
        quarantined_dates = pd.to_datetime(
            quarantine.loc[quarantine["instrument"].eq(symbol), "factor_change_date"]
        ).to_numpy(dtype="datetime64[ns]")
        action_by_symbol[symbol] = np.unique(
            np.concatenate([accepted_dates, quarantined_dates])
        )

    role_cutoffs = {
        key: pd.Timestamp(value).tz_localize("Asia/Shanghai")
        for key, value in config["label_contract"]["role_cutoffs"].items()
    }
    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    if staging.exists() or staging.is_symlink():
        raise CourageStrictC3LabelError("unsafe staging path")
    (staging / "samples").mkdir(parents=True)
    start = config["record_read_interval"][0]
    end = config["record_read_interval"][1]

    def build(symbol: str) -> dict[str, Any]:
        item = minute_records[symbol]
        source_path = Path(item["absolute_path"])
        if source_path.is_symlink() or not source_path.is_file():
            raise CourageStrictC3LabelError(f"unsafe minute source: {symbol}")
        if sha256_file_v1(source_path) != item["sha256"]:
            raise CourageStrictC3LabelError(f"minute source SHA drift: {symbol}")
        bars = (
            pq.read_table(
                source_path,
                columns=[
                    "ts_code",
                    "open",
                    "high",
                    "low",
                    "close",
                    "vol",
                    "amount",
                    "trade_date",
                    "trade_time",
                ],
                filters=[
                    ("trade_date", ">=", pd.Timestamp(start)),
                    ("trade_date", "<", pd.Timestamp(end)),
                ],
            )
            .to_pandas()
            .reset_index(drop=False)
        )
        if not bars.empty and bars["ts_code"].astype(str).ne(symbol).any():
            raise CourageStrictC3LabelError(f"minute symbol drift: {symbol}")
        frame, evidence = build_symbol_sample_partition_v1(
            instrument=symbol,
            membership=membership_by_symbol[symbol],
            grid=grid,
            axis=axis,
            bars=bars,
            industry=industry_by_symbol[symbol],
            daily_status=status_by_symbol[symbol],
            action_dates=action_by_symbol[symbol],
            role_cutoffs=role_cutoffs,
        )
        relative = f"samples/{symbol}.parquet"
        output = staging / relative
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(table, output, compression="zstd", row_group_size=8192)
        return {
            "path": relative,
            "sha256": sha256_file_v1(output),
            "bytes": output.stat().st_size,
            "rows": evidence["rows"],
            "instrument": symbol,
            "source_sha256": item["sha256"],
            "valid_counts": evidence["valid_counts"],
            "status_counts": evidence["status_counts"],
            "unknown_industry_rows": evidence["unknown_industry_rows"],
        }

    try:
        records: list[dict[str, Any]] = []
        workers = int(config["resources"]["workers"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for batch_start in range(0, len(symbols), workers * 2):
                batch = symbols[batch_start : batch_start + workers * 2]
                futures = [pool.submit(build, symbol) for symbol in batch]
                for future in concurrent.futures.as_completed(futures):
                    records.append(future.result())
                completed = min(batch_start + len(batch), len(symbols))
                print(f"C3-B labels {completed}/{len(symbols)}", flush=True)
        records.sort(key=lambda value: value["path"])
        total_rows = sum(item["rows"] for item in records)
        if total_rows != config["expected_counts"]["samples"]:
            raise CourageStrictC3LabelError("materialized sample total drift")
        valid_counts = {
            str(horizon): sum(item["valid_counts"][str(horizon)] for item in records)
            for horizon in HORIZONS
        }
        if any(value <= 0 or value > total_rows for value in valid_counts.values()):
            raise CourageStrictC3LabelError("invalid head coverage")
        unknown_industry_rows = sum(item["unknown_industry_rows"] for item in records)
        if unknown_industry_rows != config["expected_counts"]["unknown_industry_rows"]:
            raise CourageStrictC3LabelError("UNKNOWN industry row count drift")
        manifest_records = [
            {
                key: item[key]
                for key in ("path", "sha256", "bytes", "rows", "instrument")
            }
            for item in records
        ]
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "decision": "PASS_C3_B_LABEL_AND_SAMPLE_CATALOG",
            "terminal_state": "STOP_AFTER_C3_B_BEFORE_C3_FINAL_ACCEPTANCE",
            "authorization_consumed": True,
            "authorization_sha256": auth_sha,
            "config_sha256": config_sha,
            "label_contract": config["label_contract"],
            "coverage": {
                "sample_partitions": len(records),
                "samples": total_rows,
                "horizons": list(HORIZONS),
                "valid_rows_by_horizon": valid_counts,
                "unknown_industry_rows": unknown_industry_rows,
            },
            "partition_set_sha256": _canonical_root(manifest_records),
            "files": manifest_records,
            "minute_source_files_read": len(records),
            "record_read_interval": config["record_read_interval"],
            "development_test_read": False,
            "reserved_confirm_read": False,
            "holdout_read": False,
            "feature_sequence_runtime_training_selection_executed": False,
            "training_executed": False,
            "remote_push_executed": False,
        }
        manifest_path = staging / "_c3_label_manifest.json"
        manifest_path.write_bytes(_canonical_json(manifest))
        if target.exists():
            raise CourageStrictC3LabelError("immutable target appeared during build")
        os.replace(staging, target)
        return target / "_c3_label_manifest.json"
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    path = run_courage_strict_c3_labels_v1(
        project_root=args.project_root,
        config_path=args.config,
        authorization_path=args.authorization,
    )
    print(path)


if __name__ == "__main__":
    main()

"""Materialize the Courage-strict C4 mainboard feature/sequence store.

This stage consumes only the accepted C2/C3 evidence.  It computes the stock
features for the complete admitted SSE/SZSE mainboard before applying the C3
turnover membership, computes PIT sector aggregates on that complete universe,
and publishes dense official-minute shards only for symbols used by C3.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import ipc

from qlib.contrib.data.courage_strict_v1.minute_courage_strict_c2_identity_v1 import (
    sha256_file_v1,
)
from qlib.contrib.data.courage_strict_v1.minute_courage_strict_c4_features_v1 import (
    DYNAMIC_FEATURES,
    SECTOR_FEATURES,
    SLOW_FEATURES,
    STOCK_FEATURES,
    build_slow_context_v1,
    compute_sector_features_day_v1,
)
from qlib.contrib.data.courage_strict_v1.minute_courage_strict_c4_golden_v1 import (
    OPERATOR_STATEMENT,
    _load_symbol_features,
    _minute_record_map,
)


class CourageStrictC4SequenceError(RuntimeError):
    """Raised when the C4 full materialization fails closed."""


CONFIG_SCHEMA: Final[str] = "courage_strict_c4_sequence_config_v1"
AUTH_SCHEMA: Final[str] = "courage_strict_c4_sequence_authorization_v1"
CATALOG_SCHEMA: Final[str] = "courage_strict_c4_sequence_catalog_v1"
TRUE_AUTHORITIES: Final[set[str]] = {
    "execution_authorized",
    "accepted_C2_C3_read_authorized",
    "full_feature_materialization_authorized",
    "sequence_build_authorized",
    "fixed_partition_audit_authorized",
    "catalog_aggregation_authorized",
    "idempotent_reuse_verification_authorized",
}
FALSE_AUTHORITIES: Final[set[str]] = {
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


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CourageStrictC4SequenceError(f"unsafe or missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CourageStrictC4SequenceError(f"{label} must be an object")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if path.is_symlink() or not path.is_file():
        raise CourageStrictC4SequenceError(f"unsafe or missing input: {value}")
    return path


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _record(path: Path, *, base: Path, kind: str, rows: int) -> dict[str, Any]:
    return {
        "path": path.relative_to(base).as_posix(),
        "kind": kind,
        "rows": int(rows),
        "bytes": path.stat().st_size,
        "sha256": sha256_file_v1(path),
    }


def _industry_code_map(
    frame: pd.DataFrame,
) -> dict[tuple[str, pd.Timestamp], str | None]:
    keys = frame.loc[:, ["instrument", "session_date"]]
    if keys.duplicated().any():
        raise CourageStrictC4SequenceError("daily industry key is not unique")
    result: dict[tuple[str, pd.Timestamp], str | None] = {}
    for row in frame.itertuples(index=False):
        raw = row.sector_level2_code
        code = str(raw) if isinstance(raw, str) and raw else None
        result[(str(row.instrument), pd.Timestamp(row.session_date))] = code
    return result


def _validate_controls(
    root: Path, config_path: Path, auth_path: Path
) -> dict[str, Any]:
    config = _load_json(config_path, "C4 sequence config")
    auth = _load_json(auth_path, "C4 sequence authorization")
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("status") != (
        "FROZEN_NOT_AUTHORIZED"
    ):
        raise CourageStrictC4SequenceError("config identity drift")
    if any(value is not False for value in config.get("authority_matrix", {}).values()):
        raise CourageStrictC4SequenceError("config authority drift")
    for label, identity in config["inputs"].items():
        if sha256_file_v1(_resolve(root, identity["path"])) != identity["sha256"]:
            raise CourageStrictC4SequenceError(f"input drift: {label}")
    if auth.get("schema_version") != AUTH_SCHEMA or auth.get("operator_statement") != (
        OPERATOR_STATEMENT
    ):
        raise CourageStrictC4SequenceError("authorization identity drift")
    if (
        auth.get("operator_statement_sha256")
        != hashlib.sha256(OPERATOR_STATEMENT.encode()).hexdigest()
    ):
        raise CourageStrictC4SequenceError("operator statement SHA drift")
    if auth.get("config") != {
        "path": config_path.relative_to(root).as_posix(),
        "sha256": sha256_file_v1(config_path),
    }:
        raise CourageStrictC4SequenceError("authorization config drift")
    runner = Path(__file__).resolve()
    expected_runners = {
        "sequence": {
            "path": runner.relative_to(root).as_posix(),
            "sha256": sha256_file_v1(runner),
        },
        "feature_kernel": {
            "path": "qlib/contrib/data/courage_strict_v1/minute_courage_strict_c4_features_v1.py",
            "sha256": sha256_file_v1(
                root
                / "qlib/contrib/data/courage_strict_v1/minute_courage_strict_c4_features_v1.py"
            ),
        },
    }
    if auth.get("runners") != expected_runners:
        raise CourageStrictC4SequenceError("authorization runner closure drift")
    authorities = auth.get("authorities")
    if not isinstance(authorities, dict) or set(authorities) != (
        TRUE_AUTHORITIES | FALSE_AUTHORITIES
    ):
        raise CourageStrictC4SequenceError("authorization authority keys drift")
    if any(authorities[key] is not True for key in TRUE_AUTHORITIES) or any(
        authorities[key] is not False for key in FALSE_AUTHORITIES
    ):
        raise CourageStrictC4SequenceError("authorization scope drift")
    if (
        auth.get("one_time_authorization") is not True
        or auth.get("consumed_when_catalog_written") is not True
    ):
        raise CourageStrictC4SequenceError("authorization lifecycle drift")
    return config


def _axis_frame(path: Path, *, start: str, end: str) -> pd.DataFrame:
    axis = pd.read_parquet(path)
    axis = axis.loc[axis["exchange"].eq("SSE")].copy()
    axis["session_date"] = pd.to_datetime(axis["session_date"]).dt.normalize()
    axis = axis.loc[
        axis["session_date"].ge(pd.Timestamp(start))
        & axis["session_date"].lt(pd.Timestamp(end))
    ].sort_values(["session_date", "minute_slot"], kind="mergesort")
    axis = axis.reset_index(drop=True)
    if axis.groupby("session_date").size().ne(240).any():
        raise CourageStrictC4SequenceError("official axis is not 240 slots per session")
    axis["global_minute_offset"] = np.arange(len(axis), dtype=np.int64)
    axis["minute_window_end"] = pd.to_datetime(
        axis["minute_window_end"], utc=True
    ).dt.tz_convert("Asia/Shanghai")
    axis["feature_ready_time"] = pd.to_datetime(
        axis["feature_ready_time"], utc=True
    ).dt.tz_convert("Asia/Shanghai")
    if (
        not axis["feature_ready_time"]
        .eq(axis["minute_window_end"] + pd.Timedelta(minutes=1))
        .all()
    ):
        raise CourageStrictC4SequenceError("feature ready clock drift")
    return axis


def _write_axis(path: Path, axis: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(
        axis[
            [
                "global_minute_offset",
                "session_date",
                "minute_slot",
                "session",
                "minute_in_session",
                "minute_window_start",
                "minute_window_end",
                "feature_ready_time",
            ]
        ],
        preserve_index=False,
    )
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as stream:
        with ipc.new_file(stream, table.schema) as writer:
            writer.write_table(table, max_chunksize=240)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_dynamic(
    *,
    path: Path,
    axis: pd.DataFrame,
    values: np.ndarray,
    available: np.ndarray,
    missing: np.ndarray,
) -> None:
    if values.shape != (len(axis), len(DYNAMIC_FEATURES)):
        raise CourageStrictC4SequenceError("dynamic shape drift")
    valid = available & ~missing
    if not np.isfinite(values[valid]).all():
        raise CourageStrictC4SequenceError("valid dynamic value is non-finite")
    payload: dict[str, Any] = {
        "global_minute_offset": axis["global_minute_offset"].to_numpy(dtype=np.int64),
        "session_date": axis["session_date"],
        "minute_slot": axis["minute_slot"].to_numpy(dtype=np.int16),
        "feature_ready_time": axis["feature_ready_time"],
    }
    for index, feature in enumerate(DYNAMIC_FEATURES):
        payload[feature] = np.where(valid[:, index], values[:, index], 0.0).astype(
            np.float32
        )
        payload[f"{feature}__available"] = available[:, index]
        payload[f"{feature}__data_missing"] = missing[:, index]
    table = pa.Table.from_pydict(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    with temporary.open("xb") as stream:
        with ipc.new_file(stream, table.schema) as writer:
            writer.write_table(table, max_chunksize=240)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _verify_existing(target: Path, *, config_sha: str, auth_sha: str) -> Path:
    catalog_path = target / "_c4_sequence_catalog.json"
    catalog = _load_json(catalog_path, "existing C4 sequence catalog")
    if (
        catalog.get("schema_version") != CATALOG_SCHEMA
        or catalog.get("decision") != "PASS_C4_FEATURE_SEQUENCE_MATERIALIZATION"
        or catalog.get("config_sha256") != config_sha
        or catalog.get("authorization_sha256") != auth_sha
        or catalog.get("authorization_consumed") is not True
        or any(catalog.get("terminal_authority_matrix", {}).values())
    ):
        raise CourageStrictC4SequenceError("existing catalog identity drift")
    records = catalog.get("files")
    if not isinstance(records, list) or _canonical_sha(records) != catalog.get(
        "file_set_sha256"
    ):
        raise CourageStrictC4SequenceError("existing file set identity drift")
    expected = {"_c4_sequence_catalog.json"}
    for record in records:
        path = target / record["path"]
        expected.add(record["path"])
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256_file_v1(path) != record["sha256"]
        ):
            raise CourageStrictC4SequenceError("existing sequence file drift")
    actual = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise CourageStrictC4SequenceError("existing sequence exact tree drift")
    return catalog_path


def run_courage_strict_c4_sequence_v1(
    *, project_root: Path, config_path: Path, authorization_path: Path
) -> Path:
    root = project_root.resolve()
    config_path = config_path.resolve()
    authorization_path = authorization_path.resolve()
    config = _validate_controls(root, config_path, authorization_path)
    inputs = {
        key: _resolve(root, value["path"]) for key, value in config["inputs"].items()
    }
    golden = _load_json(inputs["c4_golden_report"], "accepted C4 golden report")
    if golden.get("decision") != "PASS_C4_GOLDEN_FEATURE_AUDIT":
        raise CourageStrictC4SequenceError("C4 golden gate is not PASS")
    labels = _load_json(inputs["c3_label_manifest"], "accepted C3 label manifest")
    if labels.get("decision") != "PASS_C3_B_LABEL_AND_SAMPLE_CATALOG":
        raise CourageStrictC4SequenceError("C3 label catalog is not PASS")

    config_sha = sha256_file_v1(config_path)
    auth_sha = sha256_file_v1(authorization_path)
    target = root / config["output_root"] / f"sequence-{auth_sha[:24]}"
    if target.exists():
        return _verify_existing(target, config_sha=config_sha, auth_sha=auth_sha)
    staging = target.parent / f".{target.name}.staging-{auth_sha[:24]}"
    if staging.is_symlink():
        raise CourageStrictC4SequenceError("unsafe staging path")
    staging.mkdir(parents=True, exist_ok=True)
    marker = staging / "_identity.json"
    marker_value = {"config_sha256": config_sha, "authorization_sha256": auth_sha}
    if marker.exists():
        if _load_json(marker, "staging identity") != marker_value:
            raise CourageStrictC4SequenceError("staging identity drift")
    else:
        _atomic_bytes(marker, _canonical_json(marker_value))

    expected = config["expected_counts"]
    scope = pd.read_parquet(inputs["mainboard_scope"])
    symbols = sorted(scope["instrument"].astype(str).unique())
    if len(symbols) != expected["mainboard_symbols"]:
        raise CourageStrictC4SequenceError("mainboard symbol count drift")
    membership = pd.read_parquet(inputs["membership"])
    selected_symbols = sorted(membership["instrument"].astype(str).unique())
    if (
        len(membership) != expected["membership_rows"]
        or len(selected_symbols) != expected["selected_symbols"]
    ):
        raise CourageStrictC4SequenceError("membership coverage drift")
    axis = _axis_frame(inputs["official_axis"], **config["source_interval"])
    if (
        axis["session_date"].nunique() != expected["source_sessions"]
        or len(axis) != expected["source_axis_rows"]
    ):
        raise CourageStrictC4SequenceError("source axis coverage drift")
    axis_path = staging / "official_axis.arrow"
    if not axis_path.exists():
        _write_axis(axis_path, axis)

    identity = _load_json(inputs["source_identity_manifest"], "source identity")
    minute_records = _minute_record_map(identity)
    if set(symbols) - set(minute_records):
        raise CourageStrictC4SequenceError("mainboard minute source missing")
    status = pd.read_parquet(inputs["normalized_status"])
    status = status.loc[status["instrument"].isin(symbols)].copy()
    status_by_symbol = {
        symbol: frame.copy()
        for symbol, frame in status.groupby("instrument", sort=True)
    }
    empty_status = status.iloc[0:0].copy()
    actions = pd.read_parquet(inputs["accepted_actions"])
    quarantine = pd.read_parquet(inputs["quarantined_actions"])
    date_start = {
        pd.Timestamp(date): int(frame["global_minute_offset"].iloc[0])
        for date, frame in axis.groupby("session_date", sort=True)
    }
    events: dict[str, np.ndarray] = {}
    quarantines: dict[str, np.ndarray] = {}
    for symbol in symbols:
        action_dates = pd.to_datetime(
            actions.loc[actions["instrument"].eq(symbol), "ex_date"]
        ).dt.normalize()
        quarantine_dates = pd.to_datetime(
            quarantine.loc[quarantine["instrument"].eq(symbol), "factor_change_date"]
        ).dt.normalize()
        accepted_offsets = [
            date_start[date] for date in action_dates if date in date_start
        ]
        quarantine_offsets = [
            date_start[date] for date in quarantine_dates if date in date_start
        ]
        events[symbol] = np.array(
            sorted(set(accepted_offsets + quarantine_offsets)), dtype=np.int64
        )
        quarantines[symbol] = np.array(quarantine_offsets, dtype=np.int64)

    length = len(axis)
    stock_values = np.zeros(
        (len(symbols), length, len(STOCK_FEATURES)), dtype=np.float32
    )
    stock_available = np.zeros_like(stock_values, dtype=bool)
    stock_missing = np.zeros_like(stock_values, dtype=bool)

    def load(item: tuple[int, str]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        index, symbol = item
        value, available, missing, _ = _load_symbol_features(
            symbol=symbol,
            item=minute_records[symbol],
            axis=axis,
            status=status_by_symbol.get(symbol, empty_status),
            event_offsets=events[symbol],
            quarantine_offsets=quarantines[symbol],
            start=config["source_interval"]["start"],
            end=config["source_interval"]["end"],
        )
        return index, value, available, missing

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(config["resources"]["source_workers"])
    ) as pool:
        for completed, result in enumerate(pool.map(load, enumerate(symbols)), start=1):
            index, value, available, missing = result
            stock_values[index] = value
            stock_available[index] = available
            stock_missing[index] = missing
            if completed % 250 == 0 or completed == len(symbols):
                print(
                    f"C4 sequence stock features {completed}/{len(symbols)}", flush=True
                )

    industry = pd.read_parquet(inputs["daily_industry"])
    industry["session_date"] = pd.to_datetime(industry["session_date"]).dt.normalize()
    industry = industry.loc[
        industry["instrument"].isin(symbols)
        & industry["session_date"].isin(axis["session_date"].unique())
    ]
    industry_lookup = _industry_code_map(industry)
    scope_indexed = scope.set_index("instrument").reindex(symbols)
    list_date = pd.to_datetime(scope_indexed["list_date"]).to_numpy(
        dtype="datetime64[D]"
    )
    delist_date = pd.to_datetime(scope_indexed["delist_date"]).to_numpy(
        dtype="datetime64[D]"
    )
    SectorState = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    sector_by_date: dict[pd.Timestamp, dict[str, SectorState]] = {}
    sector_valid_counts = {feature: 0 for feature in SECTOR_FEATURES}
    for completed, date in enumerate(sorted(date_start), start=1):
        day = np.datetime64(date.date())
        active = (list_date <= day) & (np.isnat(delist_date) | (delist_date >= day))
        codes = np.array(
            [industry_lookup.get((symbol, date), None) for symbol in symbols],
            dtype=object,
        )
        offset = date_start[date]
        result = compute_sector_features_day_v1(
            stock_values=stock_values[:, offset : offset + 240],
            stock_available=stock_available[:, offset : offset + 240],
            stock_missing=stock_missing[:, offset : offset + 240],
            sector_codes=codes,
            active_symbols=active,
            minimum_coverage=float(config["sector_gate"]["minimum_coverage"]),
            minimum_valid_count=int(config["sector_gate"]["minimum_valid_count"]),
        )
        sector_by_date[date] = result
        for output, available, missing, _ in result.values():
            accepted = available & ~missing & np.isfinite(output)
            for index, feature in enumerate(SECTOR_FEATURES):
                sector_valid_counts[feature] += int(accepted[:, index].sum())
        if completed % 40 == 0 or completed == len(date_start):
            print(
                f"C4 sequence sector states {completed}/{len(date_start)}", flush=True
            )

    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    selected_industry = _industry_code_map(
        industry.loc[industry["instrument"].isin(selected_symbols)]
    )
    dynamic_records: list[dict[str, Any]] = []

    def write_symbol(symbol: str) -> dict[str, Any]:
        path = staging / "dynamic" / f"{symbol}.arrow"
        if path.exists():
            return _record(path, base=staging, kind="dynamic", rows=length) | {
                "instrument": symbol
            }
        index = symbol_index[symbol]
        values = np.zeros((length, len(DYNAMIC_FEATURES)), dtype=np.float32)
        available = np.zeros_like(values, dtype=bool)
        missing = np.zeros_like(values, dtype=bool)
        values[:, : len(STOCK_FEATURES)] = stock_values[index]
        available[:, : len(STOCK_FEATURES)] = stock_available[index]
        missing[:, : len(STOCK_FEATURES)] = stock_missing[index]
        for date, start_offset in date_start.items():
            code = selected_industry.get((symbol, date), None)
            state = sector_by_date[date].get(code) if code is not None else None
            if state is None:
                continue
            output, output_available, output_missing, _ = state
            values[start_offset : start_offset + 240, len(STOCK_FEATURES) :] = output
            available[start_offset : start_offset + 240, len(STOCK_FEATURES) :] = (
                output_available
            )
            missing[start_offset : start_offset + 240, len(STOCK_FEATURES) :] = (
                output_missing
            )
        _write_dynamic(
            path=path, axis=axis, values=values, available=available, missing=missing
        )
        return _record(path, base=staging, kind="dynamic", rows=length) | {
            "instrument": symbol
        }

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(config["resources"]["write_workers"])
    ) as pool:
        for completed, record in enumerate(
            pool.map(write_symbol, selected_symbols), start=1
        ):
            dynamic_records.append(record)
            if completed % 250 == 0 or completed == len(selected_symbols):
                print(
                    f"C4 sequence dynamic shards {completed}/{len(selected_symbols)}",
                    flush=True,
                )

    daily = pd.read_parquet(inputs["normalized_daily"])
    slow = build_slow_context_v1(
        membership=membership,
        daily=daily.loc[daily["instrument"].isin(selected_symbols)],
        official_dates=pd.DatetimeIndex(axis["session_date"].unique()),
    )
    if (
        len(slow) != len(membership)
        or slow.duplicated(["instrument", "signal_date"]).any()
    ):
        raise CourageStrictC4SequenceError("slow context coverage drift")
    slow_path = staging / "slow_context.parquet"
    pq.write_table(
        pa.Table.from_pandas(slow, preserve_index=False),
        slow_path,
        compression="zstd",
        use_dictionary=True,
    )

    records = sorted(
        [
            _record(axis_path, base=staging, kind="official_axis", rows=len(axis)),
            _record(slow_path, base=staging, kind="slow_context", rows=len(slow)),
            *dynamic_records,
        ],
        key=lambda value: value["path"],
    )
    if len(records) != expected["selected_symbols"] + 2:
        raise CourageStrictC4SequenceError("sequence partition count drift")
    catalog = {
        "schema_version": CATALOG_SCHEMA,
        "decision": "PASS_C4_FEATURE_SEQUENCE_MATERIALIZATION",
        "terminal_state": "STOP_AFTER_C4_BEFORE_C5",
        "config_sha256": config_sha,
        "authorization_sha256": auth_sha,
        "authorization_consumed": True,
        "scope": {
            "security_scope": "SSE_SZSE_MAINBOARD_ONLY",
            "source_interval": config["source_interval"],
            "signal_interval": config["signal_interval"],
            "development_test_read": False,
            "reserved_confirm_read": False,
            "holdout_read": False,
        },
        "feature_identity": {
            "dynamic_order": list(DYNAMIC_FEATURES),
            "slow_order": list(SLOW_FEATURES),
            "industry_embedding": "effective_dated_sector_level2_code_or_UNKNOWN",
            "golden_report_sha256": sha256_file_v1(inputs["c4_golden_report"]),
            "golden_formula_transfer": True,
        },
        "coverage": {
            "mainboard_symbols_before_turnover_projection": len(symbols),
            "selected_symbols": len(selected_symbols),
            "membership_rows": len(membership),
            "sample_rows": labels["coverage"]["samples"],
            "source_sessions": int(axis["session_date"].nunique()),
            "source_axis_rows": len(axis),
            "dynamic_rows": len(selected_symbols) * len(axis),
            "slow_context_rows": len(slow),
            "dynamic_valid_rows_by_sector_feature": sector_valid_counts,
        },
        "sample_label_identity": {
            "path": config["inputs"]["c3_label_manifest"]["path"],
            "sha256": config["inputs"]["c3_label_manifest"]["sha256"],
            "partitions": labels["coverage"]["sample_partitions"],
            "partition_root_sha256": labels["partition_set_sha256"],
        },
        "files": records,
        "file_set_sha256": _canonical_sha(records),
        "runtime_training_selection_executed": False,
        "remote_push_executed": False,
        "terminal_authority_matrix": {
            key: False for key in sorted(TRUE_AUTHORITIES | FALSE_AUTHORITIES)
        },
    }
    catalog_path = staging / "_c4_sequence_catalog.json"
    _atomic_bytes(catalog_path, _canonical_json(catalog))
    marker.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, target)
    return _verify_existing(target, config_sha=config_sha, auth_sha=auth_sha)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    print(
        run_courage_strict_c4_sequence_v1(
            project_root=args.project_root,
            config_path=args.config,
            authorization_path=args.authorization,
        )
    )


if __name__ == "__main__":
    main()

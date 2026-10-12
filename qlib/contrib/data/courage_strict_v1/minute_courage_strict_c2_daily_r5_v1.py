"""C2-R5 daily turnover/size normalization for the Courage strict route.

This module admits only the C1 source/warm-up interval that precedes the
Development Test.  Vendor units are checked from their algebraic identities;
inconsistent fields are nulled and recorded in a quarantine table.  A daily
record becomes available only after the final official minute is ready and is
therefore forbidden for same-session intraday use.
"""

from __future__ import annotations

import argparse
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


class CourageStrictC2DailyR5Error(RuntimeError):
    """Raised when daily input cannot be normalized without silent inference."""


CONFIG_SCHEMA: Final[str] = "courage_strict_c2_daily_r5_config_v1"
AUTH_SCHEMA: Final[str] = "courage_strict_c2_daily_r5_authorization_v1"
MANIFEST_SCHEMA: Final[str] = "courage_strict_c2_daily_r5_manifest_v1"
OPERATOR_STATEMENT: Final[str] = (
    "请把下面的目标提交到著目录空间下的持久Codex.sh。确认任务已经启动并给出任务编号后再回复我。"
    "我提供给你所有权限，你需要充分利用资源，并且以较快的速度完成C0-C8任务。"
)
TRUE_AUTHORITIES: Final[set[str]] = {
    "execution_authorized",
    "source_record_read_authorized",
    "c2_daily_normalization_authorized",
    "c2_evidence_write_authorized",
}
FALSE_AUTHORITIES: Final[set[str]] = {
    "universe_build_authorized",
    "derived_dataset_materialization_authorized",
    "feature_build_authorized",
    "label_build_authorized",
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
SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "ts_code",
    "trade_date",
    "close",
    "vol",
    "turnover_rate",
    "turnover_rate_f",
    "total_share",
    "float_share",
    "free_share",
    "total_mv",
    "circ_mv",
)
NORMALIZED_COLUMNS: Final[tuple[str, ...]] = (
    "instrument",
    "session_date",
    "daily_available_at",
    "close_cny",
    "volume_lots",
    "turnover_rate_percent",
    "turnover_rate_f_percent",
    "total_share_10k_shares",
    "float_share_10k_shares",
    "free_share_10k_shares",
    "total_mv_10k_cny",
    "circ_mv_10k_cny",
    "turnover_rate_unit_valid",
    "turnover_rate_f_unit_valid",
    "total_mv_unit_valid",
    "circ_mv_unit_valid",
    "strict_t_minus_1_source_use_authorized",
    "same_session_intraday_use_authorized",
)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CourageStrictC2DailyR5Error(f"unsafe or missing {label}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CourageStrictC2DailyR5Error(f"invalid {label}") from exc
    if not isinstance(value, dict):
        raise CourageStrictC2DailyR5Error(f"{label} must be an object")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = (
        (root / value).resolve()
        if not Path(value).is_absolute()
        else Path(value).resolve()
    )
    if path.is_symlink() or not path.is_file():
        raise CourageStrictC2DailyR5Error(f"unsafe or missing input: {value}")
    return path


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _field_valid(
    actual: pd.Series,
    expected: pd.Series,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    absolute = (actual - expected).abs()
    scale = pd.concat([actual.abs(), expected.abs()], axis=1).max(axis=1)
    relative = absolute / scale.where(scale > 0, 1.0)
    finite = np.isfinite(actual) & np.isfinite(expected)
    valid = finite & (
        (absolute <= absolute_tolerance) | (relative <= relative_tolerance)
    )
    return valid.astype(bool), absolute.astype(float), relative.astype(float)


def normalize_daily_records_v1(
    source: pd.DataFrame,
    *,
    final_ready_by_date: pd.Series,
    turnover_absolute_tolerance_percent: float = 0.001,
    relative_tolerance: float = 0.005,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize units and quarantine only fields whose identities disagree."""

    missing = sorted(set(SOURCE_COLUMNS) - set(source))
    if missing:
        raise CourageStrictC2DailyR5Error(f"daily source fields missing: {missing}")
    frame = source.loc[:, list(SOURCE_COLUMNS)].copy()
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"], errors="raise"
    ).dt.normalize()
    if frame.duplicated(["ts_code", "trade_date"]).any():
        raise CourageStrictC2DailyR5Error("duplicate daily key")
    numeric = list(SOURCE_COLUMNS[2:])
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(frame[numeric].to_numpy(dtype=np.float64)).all():
        raise CourageStrictC2DailyR5Error(
            "daily source contains non-finite core values"
        )
    if (frame[["close", "total_share", "float_share", "free_share"]] <= 0).any().any():
        raise CourageStrictC2DailyR5Error(
            "daily source contains non-positive denominator"
        )

    expected = {
        "turnover_rate": frame["vol"] / frame["float_share"],
        "turnover_rate_f": frame["vol"] / frame["free_share"],
        "total_mv": frame["close"] * frame["total_share"],
        "circ_mv": frame["close"] * frame["float_share"],
    }
    validity: dict[str, pd.Series] = {}
    errors: dict[str, tuple[pd.Series, pd.Series]] = {}
    for field, expected_values in expected.items():
        absolute_tolerance = (
            turnover_absolute_tolerance_percent if field.startswith("turnover") else 0.0
        )
        valid, absolute, relative = _field_valid(
            frame[field],
            expected_values,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        validity[field] = valid
        errors[field] = (absolute, relative)

    ready = frame["trade_date"].map(final_ready_by_date)
    if ready.isna().any():
        raise CourageStrictC2DailyR5Error("daily date is absent from official axis")
    ready = pd.to_datetime(ready, utc=True).dt.tz_convert("Asia/Shanghai")
    local_date = ready.dt.tz_localize(None).dt.normalize()
    if not local_date.equals(frame["trade_date"]):
        raise CourageStrictC2DailyR5Error("daily availability date drift")

    normalized = pd.DataFrame(
        {
            "instrument": frame["ts_code"].astype(str),
            "session_date": frame["trade_date"],
            "daily_available_at": ready,
            "close_cny": frame["close"].astype(float),
            "volume_lots": frame["vol"].astype(float),
            "turnover_rate_percent": frame["turnover_rate"].where(
                validity["turnover_rate"]
            ),
            "turnover_rate_f_percent": frame["turnover_rate_f"].where(
                validity["turnover_rate_f"]
            ),
            "total_share_10k_shares": frame["total_share"].astype(float),
            "float_share_10k_shares": frame["float_share"].astype(float),
            "free_share_10k_shares": frame["free_share"].astype(float),
            "total_mv_10k_cny": frame["total_mv"].where(validity["total_mv"]),
            "circ_mv_10k_cny": frame["circ_mv"].where(validity["circ_mv"]),
            "turnover_rate_unit_valid": validity["turnover_rate"],
            "turnover_rate_f_unit_valid": validity["turnover_rate_f"],
            "total_mv_unit_valid": validity["total_mv"],
            "circ_mv_unit_valid": validity["circ_mv"],
            "strict_t_minus_1_source_use_authorized": True,
            "same_session_intraday_use_authorized": False,
        }
    )
    normalized = (
        normalized.loc[:, list(NORMALIZED_COLUMNS)]
        .sort_values(["session_date", "instrument"], kind="mergesort")
        .reset_index(drop=True)
    )

    quarantine_rows: list[dict[str, Any]] = []
    for field in expected:
        invalid = ~validity[field]
        absolute, relative = errors[field]
        for index in frame.index[invalid]:
            quarantine_rows.append(
                {
                    "instrument": str(frame.at[index, "ts_code"]),
                    "session_date": frame.at[index, "trade_date"],
                    "field": field,
                    "observed_value": float(frame.at[index, field]),
                    "identity_expected_value": float(expected[field].at[index]),
                    "absolute_error": float(absolute.at[index]),
                    "relative_error": float(relative.at[index]),
                    "field_use_authorized": False,
                    "quarantine_reason": "VENDOR_UNIT_IDENTITY_MISMATCH",
                }
            )
    quarantine = pd.DataFrame(quarantine_rows)
    if not quarantine.empty:
        quarantine = quarantine.sort_values(
            ["instrument", "session_date", "field"], kind="mergesort"
        ).reset_index(drop=True)
    return normalized, quarantine


def _validate_controls(
    *, root: Path, config_path: Path, authorization_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path, "C2-R5 config")
    auth = _load_json(authorization_path, "C2-R5 authorization")
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("status") != "FROZEN_NOT_AUTHORIZED"
    ):
        raise CourageStrictC2DailyR5Error("config identity drift")
    if config.get("record_read_interval") != ["2025-04-01", "2026-04-01"]:
        raise CourageStrictC2DailyR5Error("record interval drift")
    if any(value is not False for value in config.get("authority_matrix", {}).values()):
        raise CourageStrictC2DailyR5Error("config authorities must remain false")
    for label, identity in config["control_identities"].items():
        if sha256_file_v1(_resolve(root, identity["path"])) != identity["sha256"]:
            raise CourageStrictC2DailyR5Error(f"control drift: {label}")
    if (
        auth.get("schema_version") != AUTH_SCHEMA
        or auth.get("operator_statement") != OPERATOR_STATEMENT
    ):
        raise CourageStrictC2DailyR5Error("authorization identity drift")
    if (
        auth.get("operator_statement_sha256")
        != hashlib.sha256(OPERATOR_STATEMENT.encode()).hexdigest()
    ):
        raise CourageStrictC2DailyR5Error("operator statement SHA drift")
    if (
        auth.get("one_time_authorization") is not True
        or auth.get("consumed_when_manifest_written") is not True
    ):
        raise CourageStrictC2DailyR5Error("authorization lifecycle drift")
    if auth.get("config") != {
        "path": config_path.relative_to(root).as_posix(),
        "sha256": sha256_file_v1(config_path),
    }:
        raise CourageStrictC2DailyR5Error("authorization config drift")
    runner = Path(__file__).resolve()
    if auth.get("runner") != {
        "path": runner.relative_to(root).as_posix(),
        "sha256": sha256_file_v1(runner),
    }:
        raise CourageStrictC2DailyR5Error("authorization runner drift")
    authorities = auth.get("authorities")
    if (
        not isinstance(authorities, dict)
        or set(authorities) != TRUE_AUTHORITIES | FALSE_AUTHORITIES
    ):
        raise CourageStrictC2DailyR5Error("authorization key drift")
    if any(authorities[key] is not True for key in TRUE_AUTHORITIES) or any(
        authorities[key] is not False for key in FALSE_AUTHORITIES
    ):
        raise CourageStrictC2DailyR5Error("authorization scope drift")
    return config, auth


def run_courage_strict_c2_daily_r5_v1(
    *, project_root: Path, config_path: Path, authorization_path: Path
) -> Path:
    root = project_root.resolve()
    config_file, auth_file = config_path.resolve(), authorization_path.resolve()
    config, _ = _validate_controls(
        root=root, config_path=config_file, authorization_path=auth_file
    )
    config_sha, auth_sha = sha256_file_v1(config_file), sha256_file_v1(auth_file)
    target = root / config["output_root"] / f"authorization-{auth_sha[:24]}"
    manifest_path = target / "_daily_r5_manifest.json"
    if manifest_path.is_file():
        manifest = _load_json(manifest_path, "existing C2-R5 manifest")
        if (
            manifest.get("decision") != "PASS_C2_G09_INPUT_ADMISSION_READY"
            or manifest.get("config_sha256") != config_sha
            or manifest.get("authorization_sha256") != auth_sha
        ):
            raise CourageStrictC2DailyR5Error("existing manifest drift")
        expected_files = {item["path"]: item["sha256"] for item in manifest["files"]}
        if set(expected_files) != {
            "normalized_daily.parquet",
            "quarantined_unit_fields.parquet",
        }:
            raise CourageStrictC2DailyR5Error("existing file set drift")
        for name, digest in expected_files.items():
            if sha256_file_v1(target / name) != digest:
                raise CourageStrictC2DailyR5Error("existing file hash drift")
        return manifest_path

    source_path = _resolve(root, config["inputs"]["daily_source"]["path"])
    security_path = _resolve(root, config["inputs"]["security_master"]["path"])
    axis_path = _resolve(root, config["inputs"]["official_axis"]["path"])
    for label, path in (
        ("daily_source", source_path),
        ("security_master", security_path),
        ("official_axis", axis_path),
    ):
        if sha256_file_v1(path) != config["inputs"][label]["sha256"]:
            raise CourageStrictC2DailyR5Error(f"input drift: {label}")

    start, end = map(pd.Timestamp, config["record_read_interval"])
    security = pq.read_table(
        security_path, columns=["ts_code", "exchange", "list_date", "delist_date"]
    ).to_pandas(ignore_metadata=True)
    security["list_date"] = pd.to_datetime(security["list_date"], errors="raise")
    security["delist_date"] = pd.to_datetime(security["delist_date"], errors="coerce")
    admitted = set(
        security.loc[
            security["exchange"].isin(["SSE", "SZSE"])
            & security["list_date"].lt(end)
            & (security["delist_date"].isna() | security["delist_date"].ge(start)),
            "ts_code",
        ].astype(str)
    )
    if len(admitted) != config["expected_counts"]["admitted_instruments"]:
        raise CourageStrictC2DailyR5Error("admitted instrument count drift")

    source = pq.read_table(
        source_path,
        columns=list(SOURCE_COLUMNS),
        filters=[("trade_date", ">=", start), ("trade_date", "<", end)],
    ).to_pandas(ignore_metadata=True)
    source = source.loc[source["ts_code"].astype(str).isin(admitted)].reset_index(
        drop=True
    )
    axis = pq.read_table(
        axis_path,
        columns=["session_date", "exchange", "minute_slot", "feature_ready_time"],
    ).to_pandas(ignore_metadata=True)
    axis["session_date"] = pd.to_datetime(axis["session_date"]).dt.normalize()
    axis = axis.loc[axis["session_date"].ge(start) & axis["session_date"].lt(end)]
    final_rows = axis.loc[axis["minute_slot"].eq(239)]
    ready_counts = final_rows.groupby("session_date")["feature_ready_time"].nunique()
    if not ready_counts.eq(1).all():
        raise CourageStrictC2DailyR5Error("exchange final-ready disagreement")
    final_ready = final_rows.groupby("session_date")["feature_ready_time"].first()

    normalized, quarantine = normalize_daily_records_v1(
        source,
        final_ready_by_date=final_ready,
        turnover_absolute_tolerance_percent=config["unit_validation"][
            "turnover_absolute_tolerance_percent"
        ],
        relative_tolerance=config["unit_validation"]["relative_tolerance"],
    )
    observed = {
        "normalized_rows": len(normalized),
        "actual_instruments": normalized["instrument"].nunique(),
        "session_dates": normalized["session_date"].nunique(),
        "quarantined_field_rows": len(quarantine),
        "quarantined_record_keys": quarantine[["instrument", "session_date"]]
        .drop_duplicates()
        .shape[0],
    }
    expected = {key: config["expected_counts"][key] for key in observed}
    if observed != expected:
        raise CourageStrictC2DailyR5Error(f"normalized coverage drift: {observed}")
    if (
        normalized["same_session_intraday_use_authorized"].any()
        or not normalized["strict_t_minus_1_source_use_authorized"].all()
    ):
        raise CourageStrictC2DailyR5Error("T-1 authority drift")
    if not quarantine.empty and quarantine["field_use_authorized"].any():
        raise CourageStrictC2DailyR5Error("quarantine authority drift")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.{uuid.uuid4().hex}.staging"
    staging.mkdir(mode=0o755)
    try:
        normalized_path = staging / "normalized_daily.parquet"
        quarantine_path = staging / "quarantined_unit_fields.parquet"
        pq.write_table(
            pa.Table.from_pandas(normalized, preserve_index=False),
            normalized_path,
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pandas(quarantine, preserve_index=False),
            quarantine_path,
            compression="zstd",
        )
        files = [
            {
                "path": path.name,
                "sha256": sha256_file_v1(path),
                "bytes": path.stat().st_size,
                "rows": pq.ParquetFile(path).metadata.num_rows,
            }
            for path in (normalized_path, quarantine_path)
        ]
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "run_id": config["run_id"],
            "decision": "PASS_C2_G09_INPUT_ADMISSION_READY",
            "terminal_state": "STOP_AFTER_C2_G09_BEFORE_G10",
            "config_sha256": config_sha,
            "authorization_sha256": auth_sha,
            "authorization_consumed": True,
            "record_read_interval": config["record_read_interval"],
            "coverage": observed | {"admitted_instruments": len(admitted)},
            "unit_contract": config["unit_contract"],
            "unit_validation": config["unit_validation"],
            "availability_contract": config["availability_contract"],
            "files": files,
            "development_test_read": False,
            "reserved_confirm_read": False,
            "holdout_read": False,
            "training_executed": False,
            "remote_push_executed": False,
        }
        _write_exclusive(staging / manifest_path.name, _canonical_json(manifest))
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    path = run_courage_strict_c2_daily_r5_v1(
        project_root=args.project_root,
        config_path=args.config,
        authorization_path=args.authorization,
    )
    print(path)


if __name__ == "__main__":
    main()

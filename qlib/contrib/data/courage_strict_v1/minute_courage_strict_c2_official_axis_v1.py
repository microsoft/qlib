"""Materialize versioned official-minute-axis evidence for Courage strict C2-R1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Final

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from qlib.contrib.data.courage_strict_v1.minute_courage_strict_c2_official_axis_audit_v1 import (
    audit_official_minute_axis_v1,
)


class CourageStrictOfficialAxisError(RuntimeError):
    """Raised for unsafe, unauthorized or inconsistent C2-R1 execution."""


CONFIG_SCHEMA: Final[str] = "courage_strict_c2_official_axis_remediation_config_v1"
AUTH_SCHEMA: Final[str] = "courage_strict_c2_official_axis_remediation_authorization_v1"
MANIFEST_SCHEMA: Final[str] = "courage_strict_c2_official_axis_manifest_v1"
OPERATOR_STATEMENT: Final[str] = "按照你的建议帮我补齐"
TRUE_AUTHORITIES: Final[set[str]] = {
    "execution_authorized",
    "official_exchange_rule_source_read_authorized",
    "daily_calendar_record_read_authorized",
    "official_axis_evidence_materialization_authorized",
    "c2_remediation_evidence_write_authorized",
}
FALSE_AUTHORITIES: Final[set[str]] = {
    "raw_minute_bar_record_read_authorized",
    "security_master_record_read_authorized",
    "corporate_action_record_read_authorized",
    "daily_market_record_read_authorized",
    "industry_record_read_authorized",
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


def sha256_file_v1(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256_v1(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CourageStrictOfficialAxisError(f"unsafe or missing {label}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CourageStrictOfficialAxisError(f"invalid {label}") from exc
    if not isinstance(value, dict):
        raise CourageStrictOfficialAxisError(f"{label} must be an object")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CourageStrictOfficialAxisError(
            "configured path escapes project root"
        ) from exc
    return path


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_controls(
    *, root: Path, config_path: Path, authorization_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path, "C2-R1 config")
    authorization = _load_json(authorization_path, "C2-R1 authorization")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise CourageStrictOfficialAxisError("config schema drift")
    if config.get("status") != "FROZEN_EXECUTABLE_C2_R1_ONLY":
        raise CourageStrictOfficialAxisError("config status drift")
    if config.get("record_read_interval") != ["2025-04-01", "2026-04-01"]:
        raise CourageStrictOfficialAxisError("authorized interval drift")
    if config.get("exchanges") != ["SSE", "SZSE"]:
        raise CourageStrictOfficialAxisError("exchange scope drift")
    if config.get("timezone") != "Asia/Shanghai":
        raise CourageStrictOfficialAxisError("timezone drift")
    if config.get("exception_dates") != []:
        raise CourageStrictOfficialAxisError("unexpected session-rule exception")
    if any(value is not False for value in config.get("authority_matrix", {}).values()):
        raise CourageStrictOfficialAxisError("config authorities must remain false")

    for label, identity in config["control_identities"].items():
        path = _resolve(root, str(identity["path"]))
        if sha256_file_v1(path) != identity["sha256"]:
            raise CourageStrictOfficialAxisError(f"control identity drift: {label}")
    for record in config["exchange_rule_sources"]:
        path = _resolve(root, str(record["path"]))
        if sha256_file_v1(path) != record["sha256"]:
            raise CourageStrictOfficialAxisError("exchange-rule source drift")

    if authorization.get("schema_version") != AUTH_SCHEMA:
        raise CourageStrictOfficialAxisError("authorization schema drift")
    if authorization.get("one_time_authorization") is not True:
        raise CourageStrictOfficialAxisError("authorization must be one-time")
    if authorization.get("consumed_when_terminal_manifest_written") is not True:
        raise CourageStrictOfficialAxisError("authorization consumption drift")
    if authorization.get("operator_statement") != OPERATOR_STATEMENT:
        raise CourageStrictOfficialAxisError("operator statement drift")
    if (
        authorization.get("operator_statement_sha256")
        != hashlib.sha256(OPERATOR_STATEMENT.encode("utf-8")).hexdigest()
    ):
        raise CourageStrictOfficialAxisError("operator statement hash drift")
    expected_config = {
        "path": config_path.relative_to(root).as_posix(),
        "sha256": sha256_file_v1(config_path),
    }
    if authorization.get("config") != expected_config:
        raise CourageStrictOfficialAxisError("authorization config identity drift")
    expected_runners = {
        "materializer": {
            "path": Path(__file__).resolve().relative_to(root).as_posix(),
            "sha256": sha256_file_v1(Path(__file__).resolve()),
        },
        "independent_auditor": {
            "path": "src/aquant/research/minute_courage_strict_c2_official_axis_audit_v1.py",
            "sha256": sha256_file_v1(
                root
                / "src/aquant/research/minute_courage_strict_c2_official_axis_audit_v1.py"
            ),
        },
    }
    if authorization.get("runners") != expected_runners:
        raise CourageStrictOfficialAxisError("authorization runner identity drift")
    authorities = authorization.get("authorities")
    if not isinstance(authorities, dict) or set(authorities) != (
        TRUE_AUTHORITIES | FALSE_AUTHORITIES
    ):
        raise CourageStrictOfficialAxisError("authorization authority key drift")
    if any(authorities[key] is not True for key in TRUE_AUTHORITIES):
        raise CourageStrictOfficialAxisError("required remediation authority is false")
    if any(authorities[key] is not False for key in FALSE_AUTHORITIES):
        raise CourageStrictOfficialAxisError("forbidden remediation authority is true")
    return config, authorization


def build_official_minute_axis_v1(
    *, calendar_path: Path, config: dict[str, Any]
) -> pa.Table:
    """Generate the axis from open dates and the frozen session-rule intervals."""
    calendar = pq.read_table(
        calendar_path, columns=["trade_date", "exchange", "is_open"]
    ).to_pandas()
    calendar["trade_date"] = pd.to_datetime(calendar["trade_date"]).dt.normalize()
    start, end = map(pd.Timestamp, config["record_read_interval"])
    selected = calendar[
        (calendar["trade_date"] >= start) & (calendar["trade_date"] < end)
    ]
    if selected["trade_date"].duplicated().any():
        raise CourageStrictOfficialAxisError("daily calendar date key is not unique")
    open_dates = selected.loc[selected["is_open"].eq(1), "trade_date"].tolist()
    if len(open_dates) != int(config["expected_open_sessions"]):
        raise CourageStrictOfficialAxisError("daily calendar open-session count drift")

    rows: list[dict[str, Any]] = []
    intervals = config["session_rule"]["intervals"]
    for session_date in open_dates:
        for exchange in config["exchanges"]:
            slot = 0
            for interval in intervals:
                start_time = pd.Timestamp(
                    f"{session_date.date().isoformat()} {interval['start']}"
                ).tz_localize(config["timezone"])
                duration = int(interval["minutes"])
                for minute_in_session in range(duration):
                    window_start = start_time + pd.Timedelta(minutes=minute_in_session)
                    window_end = window_start + pd.Timedelta(minutes=1)
                    rows.append(
                        {
                            "session_date": session_date.date(),
                            "exchange": exchange,
                            "minute_slot": slot,
                            "session": interval["name"],
                            "minute_in_session": minute_in_session,
                            "minute_window_start": window_start,
                            "minute_window_end": window_end,
                            "feature_ready_time": window_end + pd.Timedelta(minutes=1),
                            "is_official_open_minute": True,
                            "rule_version": config["session_rule"]["rule_version"],
                        }
                    )
                    slot += 1
            if slot != 240:
                raise CourageStrictOfficialAxisError(
                    "session rule does not generate 240 slots"
                )

    schema = pa.schema(
        [
            pa.field("session_date", pa.date32(), nullable=False),
            pa.field("exchange", pa.string(), nullable=False),
            pa.field("minute_slot", pa.int16(), nullable=False),
            pa.field("session", pa.string(), nullable=False),
            pa.field("minute_in_session", pa.int16(), nullable=False),
            pa.field(
                "minute_window_start",
                pa.timestamp("ns", tz=config["timezone"]),
                nullable=False,
            ),
            pa.field(
                "minute_window_end",
                pa.timestamp("ns", tz=config["timezone"]),
                nullable=False,
            ),
            pa.field(
                "feature_ready_time",
                pa.timestamp("ns", tz=config["timezone"]),
                nullable=False,
            ),
            pa.field("is_official_open_minute", pa.bool_(), nullable=False),
            pa.field("rule_version", pa.string(), nullable=False),
        ]
    )
    return pa.Table.from_pylist(rows, schema=schema)


def _verify_existing(target: Path, config_sha: str, auth_sha: str) -> Path:
    terminal = target / "_official_minute_axis_manifest.json"
    value = _load_json(terminal, "existing C2-R1 terminal manifest")
    if (
        value.get("schema_version") != MANIFEST_SCHEMA
        or value.get("decision") != "PASS_C2_R1_OFFICIAL_MINUTE_AXIS"
        or value.get("config_sha256") != config_sha
        or value.get("authorization_sha256") != auth_sha
    ):
        raise CourageStrictOfficialAxisError("existing C2-R1 identity drift")
    expected_files = {
        "official_minute_axis.parquet": value["official_axis_sha256"],
        "_independent_audit.json": value["independent_audit_sha256"],
    }
    if {item.name for item in target.iterdir()} != {
        *expected_files,
        terminal.name,
    }:
        raise CourageStrictOfficialAxisError("existing C2-R1 exact file set drift")
    for name, digest in expected_files.items():
        if sha256_file_v1(target / name) != digest:
            raise CourageStrictOfficialAxisError("existing C2-R1 artifact byte drift")
    return terminal


def materialize_official_minute_axis_v1(
    *, project_root: Path, config_path: Path, authorization_path: Path
) -> Path:
    root = project_root.resolve()
    config_file = config_path.resolve()
    authorization_file = authorization_path.resolve()
    config, _ = _validate_controls(
        root=root, config_path=config_file, authorization_path=authorization_file
    )
    config_sha = sha256_file_v1(config_file)
    auth_sha = sha256_file_v1(authorization_file)
    target = _resolve(root, config["output_root"]) / f"authorization-{auth_sha[:24]}"
    if (target / "_official_minute_axis_manifest.json").is_file():
        return _verify_existing(target, config_sha, auth_sha)
    if target.exists():
        raise CourageStrictOfficialAxisError("incomplete C2-R1 target exists")

    calendar_record = config["control_identities"]["daily_calendar"]
    calendar_path = _resolve(root, calendar_record["path"])
    table = build_official_minute_axis_v1(calendar_path=calendar_path, config=config)

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    staging.mkdir()
    axis_path = staging / "official_minute_axis.parquet"
    pq.write_table(table, axis_path, compression="zstd", use_dictionary=True)
    with axis_path.open("rb") as handle:
        os.fsync(handle.fileno())

    source_records = [
        {**record, "path": _resolve(root, record["path"]).as_posix()}
        for record in config["exchange_rule_sources"]
    ]
    audit = audit_official_minute_axis_v1(
        axis_path=axis_path,
        calendar_path=calendar_path,
        source_records=source_records,
        interval=tuple(config["record_read_interval"]),
        expected_open_sessions=int(config["expected_open_sessions"]),
        expected_clock_labels_sha256=config["expected_clock_labels_sha256"],
    )
    audit_path = staging / "_independent_audit.json"
    _atomic_json(audit_path, audit)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "decision": "PASS_C2_R1_OFFICIAL_MINUTE_AXIS",
        "terminal_state": "STOP_AFTER_C2_R1_BEFORE_C2_RESTART",
        "config_path": config_file.relative_to(root).as_posix(),
        "config_sha256": config_sha,
        "authorization_path": authorization_file.relative_to(root).as_posix(),
        "authorization_sha256": auth_sha,
        "authorization_consumed": True,
        "route_sha256": config["control_identities"]["route"]["sha256"],
        "c1_sha256": config["control_identities"]["c1"]["sha256"],
        "daily_calendar": calendar_record,
        "exchange_rule_sources": config["exchange_rule_sources"],
        "session_rule": config["session_rule"],
        "record_read_interval": config["record_read_interval"],
        "official_axis_path": (target / axis_path.name).relative_to(root).as_posix(),
        "official_axis_sha256": sha256_file_v1(axis_path),
        "independent_audit_path": (target / audit_path.name)
        .relative_to(root)
        .as_posix(),
        "independent_audit_sha256": sha256_file_v1(audit_path),
        "coverage": audit,
        "raw_minute_bar_records_read": 0,
        "raw_minute_bars_used_to_infer_schedule": False,
        "development_test_read": False,
        "reserved_confirm_read": False,
        "holdout_read": False,
        "feature_label_sequence_executed": False,
        "runtime_training_selection_executed": False,
        "strategy_backtest_trading_executed": False,
        "remote_push_executed": False,
        "all_downstream_authorities_false": True,
    }
    terminal = staging / "_official_minute_axis_manifest.json"
    _atomic_json(terminal, manifest)
    os.replace(staging, target)
    return target / terminal.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    print(
        materialize_official_minute_axis_v1(
            project_root=args.project_root,
            config_path=args.config,
            authorization_path=args.authorization,
        )
    )


if __name__ == "__main__":
    main()

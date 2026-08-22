"""Bounded C2 reacceptance after the C1-v2 main-board scope amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Final

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from qlib.contrib.data.courage_strict_v1.minute_courage_strict_c2_identity_v1 import (
    sha256_file_v1,
)


class CourageStrictC2MainboardError(RuntimeError):
    """Raised when the main-board scope is not an exact accepted C2 subset."""


CONFIG_SCHEMA: Final[str] = "courage_strict_c2_mainboard_reacceptance_config_v1"
AUTH_SCHEMA: Final[str] = "courage_strict_c2_mainboard_reacceptance_authorization_v1"
REPORT_SCHEMA: Final[str] = "courage_strict_c2_mainboard_reacceptance_report_v1"
OPERATOR_STATEMENT: Final[str] = "你帮我修改一下，我就是只想研究沪深主板。"
TRUE_AUTHORITIES: Final[set[str]] = {
    "execution_authorized",
    "accepted_C2_subset_read_authorized",
    "mainboard_scope_materialization_authorized",
    "C2_mainboard_reacceptance_authorized",
    "evidence_write_authorized",
}
FALSE_AUTHORITIES: Final[set[str]] = {
    "new_external_source_acquisition_authorized",
    "turnover_pool_build_authorized",
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


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CourageStrictC2MainboardError(f"unsafe or missing {label}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CourageStrictC2MainboardError(f"invalid {label}") from exc
    if not isinstance(value, dict):
        raise CourageStrictC2MainboardError(f"{label} must be an object")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = (
        (root / value).resolve()
        if not Path(value).is_absolute()
        else Path(value).resolve()
    )
    if path.is_symlink() or not path.is_file():
        raise CourageStrictC2MainboardError(f"unsafe or missing input: {value}")
    return path


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def select_mainboard_scope_v1(security: pd.DataFrame) -> pd.DataFrame:
    required = {
        "ts_code",
        "exchange",
        "market",
        "list_date",
        "delist_date",
        "list_status",
    }
    if not required.issubset(security):
        raise CourageStrictC2MainboardError("security schema drift")
    frame = security.copy()
    frame["list_date"] = pd.to_datetime(frame["list_date"], errors="raise")
    frame["delist_date"] = pd.to_datetime(frame["delist_date"], errors="coerce")
    interval = frame.loc[
        frame["exchange"].isin(["SSE", "SZSE"])
        & frame["list_date"].lt(pd.Timestamp("2026-04-01"))
        & (
            frame["delist_date"].isna()
            | frame["delist_date"].ge(pd.Timestamp("2025-04-01"))
        )
    ].copy()
    selected = interval.loc[interval["market"].eq("主板")].copy()
    selected = selected.rename(columns={"ts_code": "instrument"})
    selected["scope_start"] = pd.Timestamp("2025-04-01")
    selected["scope_end_exclusive"] = pd.Timestamp("2026-04-01")
    selected["SSE_STAR_included"] = False
    selected["SZSE_ChiNext_included"] = False
    selected["BSE_included"] = False
    if selected["instrument"].duplicated().any():
        raise CourageStrictC2MainboardError("duplicate main-board identity")
    return selected.sort_values("instrument", kind="mergesort").reset_index(drop=True)


def exact_subset_keys_v1(
    *,
    symbols: set[str],
    daily: pd.DataFrame,
    status: pd.DataFrame,
    industry: pd.DataFrame,
) -> dict[str, Any]:
    def subset(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        required = {"instrument", "session_date"}
        if not required.issubset(frame):
            raise CourageStrictC2MainboardError(f"{label} key schema drift")
        result = frame.loc[
            frame["instrument"].astype(str).isin(symbols),
            ["instrument", "session_date"],
        ].copy()
        result["session_date"] = pd.to_datetime(result["session_date"]).dt.normalize()
        if result.duplicated().any():
            raise CourageStrictC2MainboardError(f"{label} duplicate key")
        return result.reset_index(drop=True)

    frames = [
        subset(daily, "daily"),
        subset(status, "status"),
        subset(industry, "industry"),
    ]
    if not frames[0].equals(frames[1]) or not frames[0].equals(frames[2]):
        raise CourageStrictC2MainboardError("main-board cross-input key/order drift")
    payload = "\n".join(
        f"{row.instrument}|{row.session_date.date()}" for row in frames[0].itertuples()
    ).encode()
    return {
        "rows": len(frames[0]),
        "instruments": frames[0]["instrument"].nunique(),
        "session_dates": frames[0]["session_date"].nunique(),
        "canonical_key_sha256": hashlib.sha256(payload).hexdigest(),
        "daily_status_industry_exact_key_and_order_match": True,
    }


def _validate_controls(
    *, root: Path, config_path: Path, authorization_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path, "main-board config")
    auth = _load_json(authorization_path, "main-board authorization")
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("status") != "FROZEN_NOT_AUTHORIZED"
    ):
        raise CourageStrictC2MainboardError("config identity drift")
    if any(value is not False for value in config.get("authority_matrix", {}).values()):
        raise CourageStrictC2MainboardError("config authorities must remain false")
    for label, identity in config["inputs"].items():
        if sha256_file_v1(_resolve(root, identity["path"])) != identity["sha256"]:
            raise CourageStrictC2MainboardError(f"input drift: {label}")
    if (
        auth.get("schema_version") != AUTH_SCHEMA
        or auth.get("operator_statement") != OPERATOR_STATEMENT
    ):
        raise CourageStrictC2MainboardError("authorization identity drift")
    if (
        auth.get("operator_statement_sha256")
        != hashlib.sha256(OPERATOR_STATEMENT.encode()).hexdigest()
    ):
        raise CourageStrictC2MainboardError("operator statement SHA drift")
    if (
        auth.get("one_time_authorization") is not True
        or auth.get("consumed_when_report_written") is not True
    ):
        raise CourageStrictC2MainboardError("authorization lifecycle drift")
    if auth.get("config") != {
        "path": config_path.relative_to(root).as_posix(),
        "sha256": sha256_file_v1(config_path),
    }:
        raise CourageStrictC2MainboardError("authorization config drift")
    runner = Path(__file__).resolve()
    if auth.get("runner") != {
        "path": runner.relative_to(root).as_posix(),
        "sha256": sha256_file_v1(runner),
    }:
        raise CourageStrictC2MainboardError("authorization runner drift")
    authorities = auth.get("authorities")
    if (
        not isinstance(authorities, dict)
        or set(authorities) != TRUE_AUTHORITIES | FALSE_AUTHORITIES
    ):
        raise CourageStrictC2MainboardError("authorization key drift")
    if any(authorities[key] is not True for key in TRUE_AUTHORITIES) or any(
        authorities[key] is not False for key in FALSE_AUTHORITIES
    ):
        raise CourageStrictC2MainboardError("authorization scope drift")
    return config, auth


def run_courage_strict_c2_mainboard_reacceptance_v1(
    *, project_root: Path, config_path: Path, authorization_path: Path
) -> Path:
    root = project_root.resolve()
    config_file, auth_file = config_path.resolve(), authorization_path.resolve()
    config, _ = _validate_controls(
        root=root, config_path=config_file, authorization_path=auth_file
    )
    config_sha, auth_sha = sha256_file_v1(config_file), sha256_file_v1(auth_file)
    target = root / config["output_root"] / f"audit-{auth_sha[:24]}"
    report_path = target / "_c2_mainboard_reacceptance_report.json"
    if report_path.is_file():
        report = _load_json(report_path, "existing main-board report")
        if (
            report.get("decision") != "PASS_C2_MAINBOARD_REACCEPTANCE"
            or report.get("config_sha256") != config_sha
            or report.get("authorization_sha256") != auth_sha
        ):
            raise CourageStrictC2MainboardError("existing report drift")
        scope = target / "mainboard_security_scope.parquet"
        if sha256_file_v1(scope) != report["scope_file"]["sha256"]:
            raise CourageStrictC2MainboardError("existing scope drift")
        return report_path

    paths = {
        name: _resolve(root, identity["path"])
        for name, identity in config["inputs"].items()
    }
    security = pq.read_table(paths["security_master"]).to_pandas(ignore_metadata=True)
    scope = select_mainboard_scope_v1(security)
    symbols = set(scope["instrument"])
    if len(scope) != config["expected_counts"]["mainboard_security_instruments"]:
        raise CourageStrictC2MainboardError("main-board scope count drift")
    counts = scope.groupby(["exchange", "market"]).size().to_dict()
    if counts != {("SSE", "主板"): 1715, ("SZSE", "主板"): 1499}:
        raise CourageStrictC2MainboardError("main-board exchange count drift")

    daily = pq.read_table(
        paths["normalized_daily"], columns=["instrument", "session_date"]
    ).to_pandas(ignore_metadata=True)
    status = pq.read_table(
        paths["normalized_status"], columns=["instrument", "session_date"]
    ).to_pandas(ignore_metadata=True)
    industry = pq.read_table(
        paths["daily_industry"],
        columns=["instrument", "session_date", "industry_known"],
    ).to_pandas(ignore_metadata=True)
    closure = exact_subset_keys_v1(
        symbols=symbols, daily=daily, status=status, industry=industry
    )
    if (
        closure["rows"] != config["expected_counts"]["daily_keys"]
        or closure["instruments"] != config["expected_counts"]["daily_instruments"]
        or closure["session_dates"] != config["expected_counts"]["session_dates"]
    ):
        raise CourageStrictC2MainboardError("main-board key coverage drift")
    selected_industry = industry.loc[industry["instrument"].isin(symbols)]
    industry_coverage = float(selected_industry["industry_known"].mean())
    if industry_coverage < config["minimum_industry_row_coverage"]:
        raise CourageStrictC2MainboardError("main-board industry coverage below gate")

    accepted_actions = pq.read_table(
        paths["accepted_actions"], columns=["instrument"]
    ).to_pandas(ignore_metadata=True)
    quarantined_actions = pq.read_table(
        paths["quarantined_actions"], columns=["instrument"]
    ).to_pandas(ignore_metadata=True)
    unit_quarantine = pq.read_table(
        paths["unit_quarantine"], columns=["instrument"]
    ).to_pandas(ignore_metadata=True)
    action_count = int(accepted_actions["instrument"].isin(symbols).sum())
    action_quarantine_count = int(quarantined_actions["instrument"].isin(symbols).sum())
    unit_quarantine_count = int(unit_quarantine["instrument"].isin(symbols).sum())
    expected = config["expected_counts"]
    if (action_count, action_quarantine_count, unit_quarantine_count) != (
        expected["accepted_action_rows"],
        expected["quarantined_action_rows"],
        expected["quarantined_unit_field_rows"],
    ):
        raise CourageStrictC2MainboardError("main-board quarantine/action count drift")

    identity = _load_json(paths["source_identity_manifest"], "source identity manifest")
    minute_symbols = {
        Path(record["absolute_path"]).stem
        for record in identity["files"]
        if "one_minute_OHLCVA_bars" in record.get("roles", [])
    }
    if not symbols.issubset(minute_symbols) or len(symbols & minute_symbols) != len(
        symbols
    ):
        raise CourageStrictC2MainboardError("main-board minute file closure drift")
    parent = _load_json(paths["parent_c2_report"], "parent C2 report")
    if parent.get("decision") != "PASS_C2_COMPLETE":
        raise CourageStrictC2MainboardError("parent C2 is not accepted")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.{uuid.uuid4().hex}.staging"
    staging.mkdir(mode=0o755)
    try:
        scope_path = staging / "mainboard_security_scope.parquet"
        pq.write_table(
            pa.Table.from_pandas(scope, preserve_index=False),
            scope_path,
            compression="zstd",
        )
        scope_record = {
            "path": scope_path.name,
            "sha256": sha256_file_v1(scope_path),
            "bytes": scope_path.stat().st_size,
            "rows": len(scope),
        }
        report = {
            "schema_version": REPORT_SCHEMA,
            "decision": "PASS_C2_MAINBOARD_REACCEPTANCE",
            "terminal_state": "STOP_AFTER_C2_MAINBOARD_BEFORE_C3",
            "config_sha256": config_sha,
            "authorization_sha256": auth_sha,
            "authorization_consumed": True,
            "parent_C2_decision": "PASS_C2_COMPLETE",
            "scope_contract": config["scope_contract"],
            "coverage": {
                "mainboard_security_instruments": len(scope),
                "SSE_mainboard": counts[("SSE", "主板")],
                "SZSE_mainboard": counts[("SZSE", "主板")],
                "daily_key_closure": closure,
                "industry_known_rows": int(selected_industry["industry_known"].sum()),
                "industry_row_coverage": industry_coverage,
                "minute_files": len(symbols & minute_symbols),
                "accepted_action_rows": action_count,
                "quarantined_action_rows": action_quarantine_count,
                "quarantined_unit_field_rows": unit_quarantine_count,
            },
            "scope_file": scope_record,
            "all_parent_input_hashes_reverified": True,
            "development_test_read": False,
            "reserved_confirm_read": False,
            "holdout_read": False,
            "training_executed": False,
            "remote_push_executed": False,
            "terminal_authority_matrix": {
                key: False for key in sorted(TRUE_AUTHORITIES | FALSE_AUTHORITIES)
            },
        }
        report_tmp = staging / report_path.name
        report_tmp.write_bytes(_canonical_json(report))
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    print(
        run_courage_strict_c2_mainboard_reacceptance_v1(
            project_root=args.project_root,
            config_path=args.config,
            authorization_path=args.authorization,
        )
    )


if __name__ == "__main__":
    main()

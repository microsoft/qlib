"""C2-R4 fail-closed admission for verified and unexplained factor changes."""

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


class CourageStrictC2CorporateActionR4Error(RuntimeError):
    """Raised when R3 evidence cannot be admitted without silent inference."""


CONFIG_SCHEMA: Final[str] = "courage_strict_c2_corporate_action_r4_config_v1"
AUTH_SCHEMA: Final[str] = "courage_strict_c2_corporate_action_r4_authorization_v1"
MANIFEST_SCHEMA: Final[str] = "courage_strict_c2_corporate_action_r4_manifest_v1"
OPERATOR_STATEMENT: Final[str] = (
    "请把下面的目标提交到著目录空间下的持久Codex.sh。确认任务已经启动并给出任务编号后再回复我。"
    "我提供给你所有权限，你需要充分利用资源，并且以较快的速度完成C0-C8任务。"
)
TRUE_AUTHORITIES: Final[set[str]] = {
    "execution_authorized",
    "source_record_read_authorized",
    "c2_corporate_action_fail_closed_admission_authorized",
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
QUARANTINE_COLUMNS: Final[tuple[str, ...]] = (
    "instrument",
    "factor_change_date",
    "factor_event_time",
    "observed_adjustment_factor",
    "previous_adjustment_factor",
    "factor_ratio",
    "quarantine_reason",
    "adjustment_factor_use_authorized",
    "canonical_action_inference_authorized",
    "label_head_crossing_event_valid",
    "feature_history_reset_required",
    "minimum_post_event_official_minutes_before_dynamic_use",
)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CourageStrictC2CorporateActionR4Error(f"unsafe or missing {label}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CourageStrictC2CorporateActionR4Error(f"invalid {label}") from exc
    if not isinstance(value, dict):
        raise CourageStrictC2CorporateActionR4Error(f"{label} must be an object")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CourageStrictC2CorporateActionR4Error(
            "configured path escapes project"
        ) from exc
    return path


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_controls(
    *, root: Path, config_path: Path, authorization_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path, "C2-R4 config")
    auth = _load_json(authorization_path, "C2-R4 authorization")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise CourageStrictC2CorporateActionR4Error("config schema drift")
    if config.get("status") != "FROZEN_NOT_AUTHORIZED":
        raise CourageStrictC2CorporateActionR4Error("config status drift")
    if config.get("record_read_interval") != ["2025-04-01", "2026-04-01"]:
        raise CourageStrictC2CorporateActionR4Error("record interval drift")
    if any(value is not False for value in config.get("authority_matrix", {}).values()):
        raise CourageStrictC2CorporateActionR4Error(
            "config authorities must remain false"
        )
    for label, identity in config["control_identities"].items():
        if sha256_file_v1(_resolve(root, identity["path"])) != identity["sha256"]:
            raise CourageStrictC2CorporateActionR4Error(f"control drift: {label}")
    if auth.get("schema_version") != AUTH_SCHEMA:
        raise CourageStrictC2CorporateActionR4Error("authorization schema drift")
    if auth.get("operator_statement") != OPERATOR_STATEMENT:
        raise CourageStrictC2CorporateActionR4Error("operator statement drift")
    if (
        auth.get("operator_statement_sha256")
        != hashlib.sha256(OPERATOR_STATEMENT.encode()).hexdigest()
    ):
        raise CourageStrictC2CorporateActionR4Error("operator statement SHA drift")
    if auth.get("one_time_authorization") is not True:
        raise CourageStrictC2CorporateActionR4Error("authorization must be one-time")
    if auth.get("consumed_when_manifest_written") is not True:
        raise CourageStrictC2CorporateActionR4Error("consumption rule drift")
    if auth.get("config") != {
        "path": config_path.relative_to(root).as_posix(),
        "sha256": sha256_file_v1(config_path),
    }:
        raise CourageStrictC2CorporateActionR4Error("authorization config drift")
    runner = Path(__file__).resolve()
    if auth.get("runner") != {
        "path": runner.relative_to(root).as_posix(),
        "sha256": sha256_file_v1(runner),
    }:
        raise CourageStrictC2CorporateActionR4Error("authorization runner drift")
    authorities = auth.get("authorities")
    if not isinstance(authorities, dict) or set(authorities) != (
        TRUE_AUTHORITIES | FALSE_AUTHORITIES
    ):
        raise CourageStrictC2CorporateActionR4Error("authorization key drift")
    if any(authorities[key] is not True for key in TRUE_AUTHORITIES):
        raise CourageStrictC2CorporateActionR4Error("required authority false")
    if any(authorities[key] is not False for key in FALSE_AUTHORITIES):
        raise CourageStrictC2CorporateActionR4Error("forbidden authority true")
    return config, auth


def build_quarantine_v1(unmatched: pd.DataFrame) -> pd.DataFrame:
    required = {
        "symbol",
        "trade_date",
        "event_time",
        "adj_factor",
        "previous_adjustment_factor",
        "factor_ratio",
    }
    missing = sorted(required - set(unmatched))
    if missing:
        raise CourageStrictC2CorporateActionR4Error(
            f"unmatched fields missing: {missing}"
        )
    result = pd.DataFrame(
        {
            "instrument": unmatched["symbol"].astype(str),
            "factor_change_date": pd.to_datetime(
                unmatched["trade_date"]
            ).dt.normalize(),
            "factor_event_time": unmatched["event_time"],
            "observed_adjustment_factor": unmatched["adj_factor"].astype(float),
            "previous_adjustment_factor": unmatched[
                "previous_adjustment_factor"
            ].astype(float),
            "factor_ratio": unmatched["factor_ratio"].astype(float),
            "quarantine_reason": "NO_EXACT_OFFICIAL_CORPORATE_ACTION_MATCH",
            "adjustment_factor_use_authorized": False,
            "canonical_action_inference_authorized": False,
            "label_head_crossing_event_valid": False,
            "feature_history_reset_required": True,
            "minimum_post_event_official_minutes_before_dynamic_use": 1200,
        }
    )
    if result.duplicated(["instrument", "factor_change_date"]).any():
        raise CourageStrictC2CorporateActionR4Error("duplicate quarantine key")
    return (
        result.loc[:, list(QUARANTINE_COLUMNS)]
        .sort_values(["instrument", "factor_change_date"], kind="mergesort")
        .reset_index(drop=True)
    )


def validate_partition_v1(
    *, factor_changes: pd.DataFrame, actions: pd.DataFrame, quarantine: pd.DataFrame
) -> dict[str, int]:
    def keys(frame: pd.DataFrame, symbol: str, date: str) -> list[tuple[str, str]]:
        dates = pd.to_datetime(frame[date], errors="raise").dt.strftime("%Y-%m-%d")
        return list(zip(frame[symbol].astype(str), dates, strict=True))

    action_keys = keys(actions, "instrument", "ex_date")
    quarantine_keys = keys(quarantine, "instrument", "factor_change_date")
    factor_keys = keys(factor_changes, "symbol", "trade_date")
    key_groups = (action_keys, quarantine_keys, factor_keys)
    if any(len(values) != len(set(values)) for values in key_groups):
        raise CourageStrictC2CorporateActionR4Error("duplicate partition key")
    action_set, quarantine_set, factor_set = map(
        set, (action_keys, quarantine_keys, factor_keys)
    )
    if action_set & quarantine_set:
        raise CourageStrictC2CorporateActionR4Error("accepted/quarantine overlap")
    union = action_set | quarantine_set
    if (
        len(action_keys) + len(quarantine_keys) != len(factor_keys)
        or union != factor_set
    ):
        raise CourageStrictC2CorporateActionR4Error(
            "factor partition is not exhaustive"
        )
    false_columns = [
        "adjustment_factor_use_authorized",
        "canonical_action_inference_authorized",
        "label_head_crossing_event_valid",
    ]
    if any(quarantine[column].ne(False).any() for column in false_columns):
        raise CourageStrictC2CorporateActionR4Error("quarantine fail-closed flag drift")
    if quarantine["feature_history_reset_required"].ne(True).any():
        raise CourageStrictC2CorporateActionR4Error("feature reset flag drift")
    if (
        quarantine["minimum_post_event_official_minutes_before_dynamic_use"]
        .ne(1200)
        .any()
    ):
        raise CourageStrictC2CorporateActionR4Error("lookback reset drift")
    return {
        "factor_change_rows": len(factor_changes),
        "accepted_action_rows": len(actions),
        "quarantined_factor_change_rows": len(quarantine),
        "classified_rows": len(action_keys) + len(quarantine_keys),
    }


def run_courage_strict_c2_corporate_action_r4_v1(
    *, project_root: Path, config_path: Path, authorization_path: Path
) -> Path:
    root = project_root.resolve()
    config_file = config_path.resolve()
    auth_file = authorization_path.resolve()
    config, auth = _validate_controls(
        root=root, config_path=config_file, authorization_path=auth_file
    )
    config_sha, auth_sha = sha256_file_v1(config_file), sha256_file_v1(auth_file)
    target = _resolve(root, config["output_root"]) / f"authorization-{auth_sha[:24]}"
    manifest_path = target / "_corporate_action_r4_manifest.json"
    if manifest_path.is_file():
        manifest = _load_json(manifest_path, "existing R4 manifest")
        if (
            manifest.get("decision") != "PASS_C2_G07_INPUT_ADMISSION_READY"
            or manifest.get("config_sha256") != config_sha
            or manifest.get("authorization_sha256") != auth_sha
        ):
            raise CourageStrictC2CorporateActionR4Error("existing manifest drift")
        for item in manifest["files"]:
            if sha256_file_v1(target / item["path"]) != item["sha256"]:
                raise CourageStrictC2CorporateActionR4Error("existing file drift")
        return manifest_path
    if target.exists():
        raise CourageStrictC2CorporateActionR4Error("incomplete target exists")
    r3_manifest_path = _resolve(
        root, config["control_identities"]["r3_manifest"]["path"]
    )
    r3 = _load_json(r3_manifest_path, "R3 terminal manifest")
    if r3.get("decision") != "FAIL_C2_G07_COVERAGE_DRIFT":
        raise CourageStrictC2CorporateActionR4Error("R3 decision drift")
    records = {item["path"]: item for item in r3["files"]}
    for item in records.values():
        if sha256_file_v1(r3_manifest_path.parent / item["path"]) != item["sha256"]:
            raise CourageStrictC2CorporateActionR4Error("R3 file drift")
    factor_changes = pd.read_parquet(r3_manifest_path.parent / "factor_changes.parquet")
    actions = pd.read_parquet(
        r3_manifest_path.parent / "effective_dated_corporate_actions.parquet"
    )
    unmatched = pd.read_parquet(
        r3_manifest_path.parent / "unmatched_factor_changes.parquet"
    )
    quarantine = build_quarantine_v1(unmatched)
    coverage = validate_partition_v1(
        factor_changes=factor_changes, actions=actions, quarantine=quarantine
    )
    if coverage != config["expected_coverage"]:
        raise CourageStrictC2CorporateActionR4Error("coverage drift")
    staging = target.with_name(f".{target.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    try:
        action_path = staging / "accepted_effective_dated_corporate_actions.parquet"
        quarantine_path = staging / "quarantined_factor_changes.parquet"
        pq.write_table(pa.Table.from_pandas(actions, preserve_index=False), action_path)
        pq.write_table(
            pa.Table.from_pandas(quarantine, preserve_index=False), quarantine_path
        )
        files = [
            {
                "path": path.name,
                "sha256": sha256_file_v1(path),
                "bytes": path.stat().st_size,
                "rows": pq.ParquetFile(path).metadata.num_rows,
            }
            for path in (action_path, quarantine_path)
        ]
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "decision": "PASS_C2_G07_INPUT_ADMISSION_READY",
            "terminal_state": "STOP_AFTER_R4_BEFORE_G07_REAUDIT",
            "config_sha256": config_sha,
            "authorization_sha256": auth_sha,
            "authorization_consumed": True,
            "operator_statement_sha256": auth["operator_statement_sha256"],
            "r3_manifest_sha256": sha256_file_v1(r3_manifest_path),
            "coverage": coverage,
            "partition_policy": {
                "accepted_actions": "all_required_PIT_fields_complete",
                "quarantined_changes": "no_action_inference_and_no_factor_use",
                "label_policy": "each_head_crossing_quarantined_event_is_invalid",
                "dynamic_policy": "history_resets_and_requires_1200_post_event_minutes",
                "membership_policy": "quarantine_does_not_retroactively_change_membership",
            },
            "files": files,
            "development_test_read": False,
            "reserved_confirm_read": False,
            "holdout_read": False,
            "downstream_executed": False,
            "all_downstream_authorities_false": True,
            "terminal_authority_matrix": {
                key: False for key in sorted(TRUE_AUTHORITIES | FALSE_AUTHORITIES)
            },
        }
        _write_exclusive(
            staging / "_corporate_action_r4_manifest.json",
            _canonical_json_bytes(manifest),
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, target)
        return manifest_path
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    print(
        run_courage_strict_c2_corporate_action_r4_v1(
            project_root=args.project_root,
            config_path=args.config,
            authorization_path=args.authorization,
        )
    )


if __name__ == "__main__":
    main()

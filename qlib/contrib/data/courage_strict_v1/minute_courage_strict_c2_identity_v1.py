"""Courage strict C2-A immutable source identity inventory.

This stage may enumerate, stat, and hash only explicitly configured roots.  It
does not import a dataframe/parquet reader and therefore cannot inspect source
records.  The resulting manifest is the sole byte identity eligible for a
later, separately authorized C2-B audit.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import stat
import uuid
from pathlib import Path
from typing import Any


class CourageStrictC2IdentityError(RuntimeError):
    """Raised when the bounded C2-A identity inventory is unsafe or drifts."""


SCHEMA_VERSION = "courage_strict_c2_source_identity_manifest_v1"
CONFIG_SCHEMA_VERSION = "courage_strict_c2_identity_config_v1"
AUTH_SCHEMA_VERSION = "courage_strict_c2_identity_authorization_v1"
DECISION = "PASS_C2_A_IMMUTABLE_SOURCE_IDENTITY"
ROLE_ORDER = (
    "official_exchange_calendar_and_minute_axis",
    "security_master",
    "effective_dated_security_status_and_daily_tradability",
    "corporate_actions_and_adjustment_factors",
    "one_minute_OHLCVA_bars",
    "daily_market_turnover_and_size",
    "effective_dated_industry_membership",
)
TRUE_AUTHORITIES = {
    "execution_authorized",
    "source_path_inventory_authorized",
    "source_byte_hashing_authorized",
    "c2_identity_evidence_write_authorized",
}
FALSE_AUTHORITIES = {
    "source_metadata_or_schema_read_authorized",
    "source_record_read_authorized",
    "source_admission_authorized",
    "c2_input_audit_authorized",
    "derived_data_materialization_authorized",
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
    with path.open("rb", buffering=1024 * 1024) as handle:
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
        raise CourageStrictC2IdentityError(f"unsafe or missing {label}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CourageStrictC2IdentityError(f"invalid {label}") from exc
    if not isinstance(result, dict):
        raise CourageStrictC2IdentityError(f"{label} must be an object")
    return result


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


def _regular_files(root: Path) -> list[Path]:
    if root.is_symlink() or not root.exists():
        raise CourageStrictC2IdentityError(f"unsafe or missing candidate root: {root}")
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise CourageStrictC2IdentityError(f"candidate is not file/directory: {root}")
    result: list[Path] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in sorted(directories):
            child = current_path / name
            if child.is_symlink():
                raise CourageStrictC2IdentityError(
                    f"symlink directory forbidden: {child}"
                )
        for name in sorted(files):
            child = current_path / name
            if child.is_symlink():
                raise CourageStrictC2IdentityError(f"symlink file forbidden: {child}")
            mode = child.stat(follow_symlinks=False).st_mode
            if not stat.S_ISREG(mode):
                raise CourageStrictC2IdentityError(
                    f"non-regular input forbidden: {child}"
                )
            result.append(child)
    if not result:
        raise CourageStrictC2IdentityError(
            f"candidate root has no regular files: {root}"
        )
    return sorted(result, key=lambda item: item.as_posix())


def _stat_identity(path: Path) -> dict[str, int]:
    value = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(value.st_mode):
        raise CourageStrictC2IdentityError(f"input ceased to be regular file: {path}")
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
    }


def _hash_record(item: tuple[str, Path, Path]) -> dict[str, Any]:
    role, root, path = item
    before = _stat_identity(path)
    digest = sha256_file_v1(path)
    after = _stat_identity(path)
    if before != after:
        raise CourageStrictC2IdentityError(f"source changed during hashing: {path}")
    relative = path.name if root.is_file() else path.relative_to(root).as_posix()
    return {
        "role": role,
        "candidate_root": root.as_posix(),
        "relative_path": relative,
        "absolute_path": path.as_posix(),
        **before,
        "sha256": digest,
    }


def _validate_config_and_authorization(
    *, project_root: Path, config_path: Path, authorization_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path, "C2-A config")
    authorization = _load_json(authorization_path, "C2-A authorization")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise CourageStrictC2IdentityError("C2-A config schema drift")
    if config.get("status") != "FROZEN_EXECUTABLE_C2_A_ONLY":
        raise CourageStrictC2IdentityError("C2-A config status drift")
    roles = config.get("candidate_roots")
    if not isinstance(roles, dict) or tuple(roles) != ROLE_ORDER:
        raise CourageStrictC2IdentityError("C2-A role order drift")
    for role, values in roles.items():
        if (
            not isinstance(values, list)
            or not values
            or len(values) != len(set(values))
        ):
            raise CourageStrictC2IdentityError(f"invalid roots for role {role}")
        for value in values:
            path = Path(value)
            if not path.is_absolute():
                raise CourageStrictC2IdentityError("candidate roots must be absolute")
    output_root = Path(str(config.get("output_root", "")))
    try:
        output_root.resolve().relative_to(project_root.resolve())
    except ValueError as exc:
        raise CourageStrictC2IdentityError(
            "output root must be inside project"
        ) from exc

    identities = config.get("control_identities", {})
    for label, record in identities.items():
        path = project_root / str(record.get("path", ""))
        if sha256_file_v1(path) != record.get("sha256"):
            raise CourageStrictC2IdentityError(f"control identity drift: {label}")

    if authorization.get("schema_version") != AUTH_SCHEMA_VERSION:
        raise CourageStrictC2IdentityError("authorization schema drift")
    if authorization.get("one_time_authorization") is not True:
        raise CourageStrictC2IdentityError("authorization must be one-time")
    if authorization.get("consumed_when_identity_manifest_written") is not True:
        raise CourageStrictC2IdentityError("authorization consumption drift")
    config_identity = authorization.get("config", {})
    runner_identity = authorization.get("runner", {})
    if config_identity != {
        "path": config_path.relative_to(project_root).as_posix(),
        "sha256": sha256_file_v1(config_path),
    }:
        raise CourageStrictC2IdentityError("authorization config identity drift")
    runner = Path(__file__).resolve()
    if runner_identity != {
        "path": runner.relative_to(project_root).as_posix(),
        "sha256": sha256_file_v1(runner),
    }:
        raise CourageStrictC2IdentityError("authorization runner identity drift")
    authorities = authorization.get("authorities")
    if not isinstance(authorities, dict):
        raise CourageStrictC2IdentityError("authorization authority matrix missing")
    if set(authorities) != TRUE_AUTHORITIES | FALSE_AUTHORITIES:
        raise CourageStrictC2IdentityError("authorization authority key drift")
    if any(authorities[key] is not True for key in TRUE_AUTHORITIES):
        raise CourageStrictC2IdentityError("required C2-A authority is false")
    if any(authorities[key] is not False for key in FALSE_AUTHORITIES):
        raise CourageStrictC2IdentityError("forbidden authority is true")
    statement = authorization.get("operator_statement")
    if not isinstance(statement, str) or not statement:
        raise CourageStrictC2IdentityError("operator statement missing")
    if hashlib.sha256(statement.encode("utf-8")).hexdigest() != authorization.get(
        "operator_statement_sha256"
    ):
        raise CourageStrictC2IdentityError("operator statement hash drift")
    return config, authorization


def _verify_existing(
    *, terminal: Path, config_sha256: str, authorization_sha256: str
) -> dict[str, Any]:
    value = _load_json(terminal, "C2-A identity manifest")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("decision") != DECISION
    ):
        raise CourageStrictC2IdentityError("existing terminal identity drift")
    if value.get("config_sha256") != config_sha256:
        raise CourageStrictC2IdentityError("existing config identity drift")
    if value.get("authorization_sha256") != authorization_sha256:
        raise CourageStrictC2IdentityError("existing authorization identity drift")
    records = value.get("files")
    if not isinstance(records, list) or len(records) != value.get("unique_file_count"):
        raise CourageStrictC2IdentityError("existing file inventory drift")
    for record in records:
        path = Path(record["absolute_path"])
        if _stat_identity(path) != {
            key: record[key] for key in ("device", "inode", "mode", "bytes", "mtime_ns")
        }:
            raise CourageStrictC2IdentityError("existing source stat drift")
        if sha256_file_v1(path) != record["sha256"]:
            raise CourageStrictC2IdentityError("existing source byte drift")
    root = canonical_sha256_v1(records)
    if root != value.get("file_set_sha256"):
        raise CourageStrictC2IdentityError("existing file-set root drift")
    return value


def materialize_courage_strict_c2_identity_v1(
    *, project_root: Path, config_path: Path, authorization_path: Path
) -> Path:
    root = project_root.resolve()
    config_file = config_path.resolve()
    authorization_file = authorization_path.resolve()
    config, _ = _validate_config_and_authorization(
        project_root=root,
        config_path=config_file,
        authorization_path=authorization_file,
    )
    config_sha256 = sha256_file_v1(config_file)
    authorization_sha256 = sha256_file_v1(authorization_file)
    run_id = canonical_sha256_v1(
        {
            "schema_version": SCHEMA_VERSION,
            "config_sha256": config_sha256,
            "authorization_sha256": authorization_sha256,
        }
    )
    target = Path(config["output_root"]).resolve() / f"identity-{run_id[:20]}"
    terminal = target / "_source_identity_manifest.json"
    if terminal.is_file():
        _verify_existing(
            terminal=terminal,
            config_sha256=config_sha256,
            authorization_sha256=authorization_sha256,
        )
        return terminal
    if target.exists():
        raise CourageStrictC2IdentityError("incomplete C2-A target exists")

    work: list[tuple[str, Path, Path]] = []
    path_roles: dict[str, list[str]] = {}
    for role, root_values in config["candidate_roots"].items():
        for root_value in root_values:
            candidate_root = Path(root_value).resolve()
            for path in _regular_files(candidate_root):
                key = path.as_posix()
                path_roles.setdefault(key, []).append(role)
                work.append((role, candidate_root, path))

    unique_work: dict[str, tuple[str, Path, Path]] = {}
    for item in work:
        unique_work.setdefault(item[2].as_posix(), item)
    workers = int(config.get("hash_workers", 16))
    if not 1 <= workers <= 64:
        raise CourageStrictC2IdentityError("hash worker count outside frozen bound")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(_hash_record, unique_work.values()))
    records.sort(key=lambda item: item["absolute_path"])
    for record in records:
        record["roles"] = sorted(set(path_roles[record["absolute_path"]]))
        record.pop("role")

    role_summaries = {}
    for role in ROLE_ORDER:
        selected = [item for item in records if role in item["roles"]]
        role_summaries[role] = {
            "files": len(selected),
            "bytes": sum(item["bytes"] for item in selected),
            "identity_root_sha256": canonical_sha256_v1(selected),
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION,
        "terminal_state": "STOP_AFTER_C2_A_BEFORE_C2_B",
        "run_id": run_id,
        "config_path": config_file.relative_to(root).as_posix(),
        "config_sha256": config_sha256,
        "authorization_path": authorization_file.relative_to(root).as_posix(),
        "authorization_sha256": authorization_sha256,
        "authorization_consumed": True,
        "record_content_parsed": False,
        "source_metadata_or_schema_read": False,
        "source_admission_executed": False,
        "derived_data_materialized": False,
        "development_test_read": False,
        "reserved_confirm_read": False,
        "holdout_read": False,
        "training_executed": False,
        "remote_push_executed": False,
        "role_summaries": role_summaries,
        "unique_file_count": len(records),
        "unique_bytes": sum(item["bytes"] for item in records),
        "file_set_sha256": canonical_sha256_v1(records),
        "files": records,
    }
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    staging.mkdir()
    _atomic_json(staging / terminal.name, manifest)
    os.replace(staging, target)
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    terminal = materialize_courage_strict_c2_identity_v1(
        project_root=args.project_root,
        config_path=args.config,
        authorization_path=args.authorization,
    )
    print(terminal.as_posix())


if __name__ == "__main__":
    main()

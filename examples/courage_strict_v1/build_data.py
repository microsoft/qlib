"""Construct Courage Strict V1 PIT, labels, features, and Qlib data products.

The historical stage configurations are semantic templates only. This command
creates new contracts and one-time authorizations bound to this Qlib-fork code,
then computes every Train/Valid data product from project-local source facts and
the canonical raw minute files. No historical learned state is read.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from qlib.contrib.data.courage_strict_v1 import (
    minute_courage_strict_c3_labels_v1 as labels,
)
from qlib.contrib.data.courage_strict_v1 import (
    minute_courage_strict_c3_membership_v1 as membership,
)
from qlib.contrib.data.courage_strict_v1 import (
    minute_courage_strict_c4_golden_v1 as golden,
)
from qlib.contrib.data.courage_strict_v1 import (
    minute_courage_strict_c4_sequence_v1 as sequence,
)
from qlib.contrib.data.courage_strict_v1 import qlib_materializer

RUN_ID = "courage_strict_v1"
CONFIG_DIR = Path("examples/courage_strict_v1/configs")
LEGACY_CONFIG_DIR = CONFIG_DIR / "legacy_semantics"
SOURCE_FACTS_DIR = Path("data/courage_strict_v1/source")
ARTIFACTS_DIR = Path("artifacts/courage_strict_v1")
RAW_MINUTE_ROOT = Path(
    "/data1/lxl/workspace/datasets/.tmp/dataset/xianyu/full_20260724/"
    "行情数据(全量至20260724）/stock_1min"
)
MINUTE_SNAPSHOT_START = "2025-04-01"
MINUTE_SNAPSHOT_END = "2026-05-01"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value))
    return path


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _identity(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _relative_identity(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(root).as_posix(),
        "sha256": _sha256(path),
    }


def _remap_inputs(
    *, root: Path, template: dict[str, Any], replacements: dict[str, Path]
) -> None:
    for label, identity in template["inputs"].items():
        if label in replacements:
            path = replacements[label]
        elif label == "c1":
            path = (
                root / CONFIG_DIR / "courage_strict_c1_implementation_decisions_v1.json"
            )
        elif label == "c1_mainboard":
            path = (
                root
                / CONFIG_DIR
                / "courage_strict_c1_mainboard_scope_amendment_v2.json"
            )
        else:
            path = root / SOURCE_FACTS_DIR / label / Path(identity["path"]).name
        template["inputs"][label] = _identity(path)


def _authorization(
    *,
    root: Path,
    schema: str,
    scope: str,
    config_path: Path,
    runner_path: Path,
    true_authorities: set[str],
    false_authorities: set[str],
    consumed_key: str,
    runners: dict[str, Path] | None = None,
) -> dict[str, Any]:
    statement = membership.OPERATOR_STATEMENT
    result: dict[str, Any] = {
        "schema_version": schema,
        "authorization_scope": scope,
        "one_time_authorization": True,
        consumed_key: True,
        "operator_statement": statement,
        "operator_statement_sha256": hashlib.sha256(statement.encode()).hexdigest(),
        "config": _relative_identity(root, config_path),
        "authorities": {
            key: key in true_authorities
            for key in sorted(true_authorities | false_authorities)
        },
    }
    if runners is None:
        result["runner"] = _relative_identity(root, runner_path)
    else:
        result["runners"] = {
            key: _relative_identity(root, path) for key, path in runners.items()
        }
    return result


def _stage_paths(root: Path, name: str) -> tuple[Path, Path]:
    directory = root / CONFIG_DIR / "runtime"
    return directory / f"{name}.json", directory / f"{name}_authorization.json"


def _validate_source_facts(root: Path) -> Path:
    """Pin the transferred source facts before constructing any derived table."""
    templates = [
        _read(root / LEGACY_CONFIG_DIR / "courage_strict_c3_membership_v1.json"),
        _read(root / LEGACY_CONFIG_DIR / "courage_strict_c4_golden_v1.json"),
    ]
    ignored = {
        "c1",
        "c1_mainboard",
        "c3_final_report",
        "membership",
        "source_identity_manifest",
    }
    expected: dict[str, dict[str, str]] = {}
    for template in templates:
        for label, identity in template["inputs"].items():
            if label in ignored:
                continue
            previous = expected.setdefault(label, identity)
            if previous["sha256"] != identity["sha256"]:
                raise RuntimeError(f"source fact identity conflict: {label}")

    records: list[dict[str, Any]] = []
    for label, identity in sorted(expected.items()):
        path = root / SOURCE_FACTS_DIR / label / Path(identity["path"]).name
        digest = _sha256(path)
        if digest != identity["sha256"]:
            raise RuntimeError(f"source fact SHA drift: {label}")
        records.append(
            {
                "label": label,
                "path": path.relative_to(root).as_posix(),
                "sha256": digest,
                "bytes": path.stat().st_size,
            }
        )
    manifest = root / SOURCE_FACTS_DIR / "_source_facts_manifest.json"
    value = {
        "schema_version": "courage_strict_v1_source_facts_manifest_v1",
        "run_id": RUN_ID,
        "decision": "PASS_V1_SOURCE_FACTS_PINNED",
        "records": records,
        "record_set_sha256": hashlib.sha256(_canonical(records)).hexdigest(),
        "old_learned_state_imported": False,
    }
    if manifest.is_file():
        if _read(manifest) != value:
            raise RuntimeError("existing source fact manifest drift")
    else:
        _write(manifest, value)
    return manifest


def _run_membership(root: Path) -> tuple[Path, Path, Path]:
    template_path = root / LEGACY_CONFIG_DIR / "courage_strict_c3_membership_v1.json"
    config = _read(template_path)
    _remap_inputs(root=root, template=config, replacements={})
    config["route_id"] = RUN_ID
    config["output_root"] = str(root / ARTIFACTS_DIR / "stages/membership")
    config_path, auth_path = _stage_paths(root, "membership")
    _write(config_path, config)
    runner = Path(membership.__file__).resolve()
    auth = _authorization(
        root=root,
        schema=membership.AUTH_SCHEMA,
        scope="V1_Qlib_fresh_T_minus_1_membership_and_signal_grid",
        config_path=config_path,
        runner_path=runner,
        true_authorities=membership.TRUE_AUTHORITIES,
        false_authorities=membership.FALSE_AUTHORITIES,
        consumed_key="consumed_when_manifest_written",
    )
    _write(auth_path, auth)
    manifest = membership.run_courage_strict_c3_membership_v1(
        project_root=root, config_path=config_path, authorization_path=auth_path
    )
    directory = manifest.parent
    return manifest, directory / "membership.parquet", directory / "signal_grid.parquet"


def _construct_minute_snapshot(root: Path) -> Path:
    """Create the project-owned minute source used by all V1 data stages."""
    scope_path = (
        root / SOURCE_FACTS_DIR / "mainboard_scope/mainboard_security_scope.parquet"
    )
    symbols = sorted(
        pd.read_parquet(scope_path, columns=["instrument"])["instrument"].unique()
    )
    if len(symbols) != 3214:
        raise RuntimeError(f"mainboard scope drift: {len(symbols)}")

    source_root = RAW_MINUTE_ROOT.resolve()
    target_root = root / SOURCE_FACTS_DIR / "minute_snapshot"
    output_root = target_root / "stock_1min"
    identity_path = (
        root
        / SOURCE_FACTS_DIR
        / "source_identity_manifest/_source_identity_manifest.json"
    )
    if identity_path.is_file():
        identity = _read(identity_path)
        records = identity.get("files")
        if not isinstance(records, list) or len(records) != len(symbols):
            raise RuntimeError("existing minute identity coverage drift")
        for item in records:
            path = Path(item["absolute_path"])
            if not path.is_file() or _sha256(path) != item["sha256"]:
                raise RuntimeError(f"existing minute snapshot drift: {path}")
        return identity_path

    staging = target_root.parent / f".minute_snapshot.staging-{uuid.uuid4().hex}"
    if staging.exists() or staging.is_symlink():
        raise RuntimeError(f"unsafe minute staging path: {staging}")
    staging_output = staging / "stock_1min"
    staging_output.mkdir(parents=True)
    columns = [
        "ts_code",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        "trade_date",
        "trade_time",
    ]

    def build(symbol: str) -> dict[str, Any]:
        source = source_root / f"{symbol}.parquet"
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"missing physical minute source: {symbol}")
        table = pq.read_table(
            source,
            columns=columns,
            filters=[
                ("trade_date", ">=", pd.Timestamp(MINUTE_SNAPSHOT_START)),
                ("trade_date", "<", pd.Timestamp(MINUTE_SNAPSHOT_END)),
            ],
        )
        output = staging_output / f"{symbol}.parquet"
        pq.write_table(
            table,
            output,
            compression="zstd",
            row_group_size=65536,
            use_dictionary=False,
        )
        return {
            "absolute_path": str((output_root / output.name).resolve()),
            "relative_path": output.name,
            "bytes": output.stat().st_size,
            "rows": table.num_rows,
            "sha256": _sha256(output),
            "roles": ["one_minute_OHLCVA_bars"],
        }

    records: list[dict[str, Any]] = []
    workers = min(48, max(8, (os.cpu_count() or 8) // 4))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for completed, record in enumerate(pool.map(build, symbols), start=1):
            records.append(record)
            if completed % 100 == 0 or completed == len(symbols):
                print(
                    f"minute snapshot {completed}/{len(symbols)} workers={workers}",
                    flush=True,
                )
    records.sort(key=lambda item: item["relative_path"])
    staging_identity = staging / "_source_identity_manifest.json"
    _write(
        staging_identity,
        {
            "schema_version": "courage_strict_v1_minute_source_identity_v1",
            "run_id": RUN_ID,
            "decision": "PASS_V1_MINUTE_SOURCE_CONSTRUCTED",
            "interval": [MINUTE_SNAPSHOT_START, MINUTE_SNAPSHOT_END],
            "symbol_count": len(symbols),
            "files": records,
            "file_set_sha256": hashlib.sha256(_canonical(records)).hexdigest(),
            "may_read": False,
            "june_read": False,
        },
    )
    if target_root.exists() or target_root.is_symlink():
        raise RuntimeError(f"minute snapshot target already exists: {target_root}")
    os.replace(staging, target_root)
    identity_path.parent.mkdir(parents=True, exist_ok=False)
    identity_path.write_bytes(
        (target_root / "_source_identity_manifest.json").read_bytes()
    )
    return identity_path


def _run_labels(
    root: Path,
    membership_manifest: Path,
    membership_path: Path,
    signal_grid: Path,
    source_identity: Path,
) -> Path:
    template_path = root / LEGACY_CONFIG_DIR / "courage_strict_c3_labels_v2.json"
    config = _read(template_path)
    _remap_inputs(
        root=root,
        template=config,
        replacements={
            "c3_membership_manifest": membership_manifest,
            "membership": membership_path,
            "signal_grid": signal_grid,
            "source_identity_manifest": source_identity,
        },
    )
    config["route_id"] = RUN_ID
    config["output_root"] = str(root / ARTIFACTS_DIR / "stages/labels")
    config_path, auth_path = _stage_paths(root, "labels")
    _write(config_path, config)
    runner = Path(labels.__file__).resolve()
    auth = _authorization(
        root=root,
        schema=labels.AUTH_SCHEMA,
        scope="V1_Qlib_fresh_Train_Valid_VWAP1_labels",
        config_path=config_path,
        runner_path=runner,
        true_authorities=labels.TRUE_AUTHORITIES,
        false_authorities=labels.FALSE_AUTHORITIES,
        consumed_key="consumed_when_manifest_written",
    )
    _write(auth_path, auth)
    return labels.run_courage_strict_c3_labels_v1(
        project_root=root, config_path=config_path, authorization_path=auth_path
    )


def _write_c3_acceptance(
    root: Path, membership_manifest: Path, label_manifest: Path
) -> Path:
    path = root / CONFIG_DIR / "runtime/c3_acceptance.json"
    return _write(
        path,
        {
            "schema_version": "courage_strict_v1_qlib_c3_acceptance_v1",
            "decision": "PASS_C3_COMPLETE",
            "route_id": RUN_ID,
            "membership_manifest": _identity(membership_manifest),
            "label_manifest": _identity(label_manifest),
            "historical_learned_state_read": False,
            "development_replay_read": False,
            "may_read": False,
            "june_read": False,
        },
    )


def _run_golden(root: Path, c3_report: Path, membership_path: Path) -> Path:
    template_path = root / LEGACY_CONFIG_DIR / "courage_strict_c4_golden_v1.json"
    config = _read(template_path)
    _remap_inputs(
        root=root,
        template=config,
        replacements={"c3_final_report": c3_report, "membership": membership_path},
    )
    config["route_id"] = RUN_ID
    config["output_root"] = str(root / ARTIFACTS_DIR / "stages/golden")
    config_path, auth_path = _stage_paths(root, "golden")
    _write(config_path, config)
    runner = Path(golden.__file__).resolve()
    kernel = Path(
        root
        / "qlib/contrib/data/courage_strict_v1/minute_courage_strict_c4_features_v1.py"
    )
    auth = _authorization(
        root=root,
        schema=golden.AUTH_SCHEMA,
        scope="V1_Qlib_fresh_2025_06_feature_golden",
        config_path=config_path,
        runner_path=runner,
        true_authorities=golden.TRUE_AUTHORITIES,
        false_authorities=golden.FALSE_AUTHORITIES,
        consumed_key="consumed_when_report_written",
        runners={"golden": runner, "feature_kernel": kernel},
    )
    _write(auth_path, auth)
    return golden.run_courage_strict_c4_golden_v1(
        project_root=root, config_path=config_path, authorization_path=auth_path
    )


def _run_sequence(
    root: Path,
    c3_report: Path,
    label_manifest: Path,
    golden_report: Path,
    membership_path: Path,
) -> Path:
    template_path = root / LEGACY_CONFIG_DIR / "courage_strict_c4_sequence_v1.json"
    config = _read(template_path)
    _remap_inputs(
        root=root,
        template=config,
        replacements={
            "c3_final_report": c3_report,
            "c3_label_manifest": label_manifest,
            "c4_golden_report": golden_report,
            "membership": membership_path,
        },
    )
    config["route_id"] = RUN_ID
    config["output_root"] = str(root / ARTIFACTS_DIR / "stages/features")
    config_path, auth_path = _stage_paths(root, "sequence")
    _write(config_path, config)
    runner = Path(sequence.__file__).resolve()
    kernel = Path(
        root
        / "qlib/contrib/data/courage_strict_v1/minute_courage_strict_c4_features_v1.py"
    )
    auth = _authorization(
        root=root,
        schema=sequence.AUTH_SCHEMA,
        scope="V1_Qlib_fresh_full_feature_materialization",
        config_path=config_path,
        runner_path=runner,
        true_authorities=sequence.TRUE_AUTHORITIES,
        false_authorities=sequence.FALSE_AUTHORITIES,
        consumed_key="consumed_when_catalog_written",
        runners={"sequence": runner, "feature_kernel": kernel},
    )
    _write(auth_path, auth)
    return sequence.run_courage_strict_c4_sequence_v1(
        project_root=root, config_path=config_path, authorization_path=auth_path
    )


def _run_qlib_provider(root: Path, feature_catalog: Path, label_manifest: Path) -> Path:
    config_path, auth_path = _stage_paths(root, "qlib_provider")
    config = {
        "schema_version": "courage_strict_v1_qlib_config_v1",
        "run_id": RUN_ID,
        "status": "FROZEN_FOR_EXECUTION",
        "stores": {
            "features": {
                "root": feature_catalog.parent.relative_to(root).as_posix(),
                "catalog_path": feature_catalog.relative_to(root).as_posix(),
                "catalog_sha256": _sha256(feature_catalog),
            },
            "labels": {
                "root": label_manifest.parent.relative_to(root).as_posix(),
                "catalog_path": label_manifest.relative_to(root).as_posix(),
                "catalog_sha256": _sha256(label_manifest),
            },
        },
        "minute_source": {
            "root": "data/courage_strict_v1/source/minute_snapshot/stock_1min",
            "identity_path": (
                "data/courage_strict_v1/source/source_identity_manifest/"
                "_source_identity_manifest.json"
            ),
            "identity_sha256": _sha256(
                root / "data/courage_strict_v1/source/source_identity_manifest/"
                "_source_identity_manifest.json"
            ),
        },
        "output_root": "data/courage_strict_v1/qlib_provider",
        "workers": min(96, max(16, (os.cpu_count() or 16) * 3 // 8)),
    }
    _write(config_path, config)
    runner = Path(qlib_materializer.__file__).resolve()
    authorities = {
        key: key in qlib_materializer.AUTH_TRUE
        for key in sorted(qlib_materializer.AUTH_TRUE | qlib_materializer.AUTH_FALSE)
    }
    auth = {
        "schema_version": "courage_strict_v1_qlib_authorization_v1",
        "one_time_authorization": True,
        "operator_statement": membership.OPERATOR_STATEMENT,
        "config": _relative_identity(root, config_path),
        "runner": _relative_identity(root, runner),
        **authorities,
    }
    _write(auth_path, auth)
    qlib_materializer.run_courage_strict_v1_qlib(
        project_root=root,
        config_path=config_path,
        authorization_path=auth_path,
    )
    return root / config["output_root"] / "_courage_strict_v1_qlib_catalog.json"


def run(root: Path) -> dict[str, Path]:
    root = root.resolve()
    source_facts_manifest = _validate_source_facts(root)
    membership_manifest, membership_path, signal_grid = _run_membership(root)
    source_identity = _construct_minute_snapshot(root)
    label_manifest = _run_labels(
        root,
        membership_manifest,
        membership_path,
        signal_grid,
        source_identity,
    )
    c3_report = _write_c3_acceptance(root, membership_manifest, label_manifest)
    golden_report = _run_golden(root, c3_report, membership_path)
    sequence_catalog = _run_sequence(
        root, c3_report, label_manifest, golden_report, membership_path
    )
    provider_catalog = _run_qlib_provider(root, sequence_catalog, label_manifest)
    return {
        "source_facts": source_facts_manifest,
        "membership": membership_manifest,
        "labels": label_manifest,
        "golden": golden_report,
        "features": sequence_catalog,
        "qlib_provider": provider_catalog,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qlib-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    for name, path in run(args.qlib_root).items():
        print(f"{name}={path}", flush=True)


if __name__ == "__main__":
    main()

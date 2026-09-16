"""C2-R6 effective-dated CNInfo industry admission for the full C1 scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import uuid
import warnings
from concurrent import futures
from pathlib import Path
from typing import Any, Final

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from qlib.contrib.data.courage_strict_v1.minute_courage_strict_c2_identity_v1 import (
    sha256_file_v1,
)


class CourageStrictC2IndustryR6Error(RuntimeError):
    """Raised when full-scope industry history cannot be admitted safely."""


CONFIG_SCHEMA: Final[str] = "courage_strict_c2_industry_r6_config_v1"
AUTH_SCHEMA: Final[str] = "courage_strict_c2_industry_r6_authorization_v1"
MANIFEST_SCHEMA: Final[str] = "courage_strict_c2_industry_r6_manifest_v1"
OPERATOR_STATEMENT: Final[str] = (
    "请把下面的目标提交到著目录空间下的持久Codex.sh。确认任务已经启动并给出任务编号后再回复我。"
    "我提供给你所有权限，你需要充分利用资源，并且以较快的速度完成C0-C8任务。"
)
TRUE_AUTHORITIES: Final[set[str]] = {
    "execution_authorized",
    "external_source_acquisition_authorized",
    "source_record_read_authorized",
    "c2_industry_normalization_authorized",
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
RAW_COLUMNS: Final[tuple[str, ...]] = (
    "instrument",
    "provider_symbol",
    "change_date",
    "classification_standard_code",
    "classification_standard_name",
    "industry_code",
    "industry_section",
    "industry_subcategory",
    "industry_major",
    "industry_medium",
)
PIT_COLUMNS: Final[tuple[str, ...]] = (
    "instrument",
    "change_date",
    "classification_standard_code",
    "industry_code",
    "sector_level2_code",
    "sector_level2_name",
)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CourageStrictC2IndustryR6Error(f"unsafe or missing {label}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CourageStrictC2IndustryR6Error(f"invalid {label}") from exc
    if not isinstance(value, dict):
        raise CourageStrictC2IndustryR6Error(f"{label} must be an object")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = (
        (root / value).resolve()
        if not Path(value).is_absolute()
        else Path(value).resolve()
    )
    if path.is_symlink() or not path.is_file():
        raise CourageStrictC2IndustryR6Error(f"unsafe or missing input: {value}")
    return path


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(_canonical_json(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def normalize_provider_history_v1(instrument: str, frame: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "证券代码": "provider_symbol",
        "变更日期": "change_date",
        "分类标准编码": "classification_standard_code",
        "分类标准": "classification_standard_name",
        "行业编码": "industry_code",
        "行业门类": "industry_section",
        "行业次类": "industry_subcategory",
        "行业大类": "industry_major",
        "行业中类": "industry_medium",
    }
    if frame.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)
    missing = sorted(set(mapping) - set(frame))
    if missing:
        raise CourageStrictC2IndustryR6Error(f"provider schema drift: {missing}")
    result = frame.loc[:, list(mapping)].rename(columns=mapping).copy()
    result.insert(0, "instrument", instrument)
    result["provider_symbol"] = result["provider_symbol"].astype(str).str.zfill(6)
    if set(result["provider_symbol"]) != {instrument[:6]}:
        raise CourageStrictC2IndustryR6Error(f"provider symbol drift: {instrument}")
    result["change_date"] = pd.to_datetime(
        result["change_date"], errors="raise"
    ).dt.normalize()
    for column in RAW_COLUMNS:
        if column != "change_date":
            result[column] = result[column].fillna("").astype(str)
    result = result.drop_duplicates().sort_values(
        ["change_date", "classification_standard_code", "industry_code"],
        kind="mergesort",
    )
    return result.loc[:, list(RAW_COLUMNS)].reset_index(drop=True)


def build_pit_history_v1(raw: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    required = {"类目编码", "类目名称", "父类编码", "分级", "行业类型编码"}
    if not required.issubset(taxonomy):
        raise CourageStrictC2IndustryR6Error("taxonomy schema drift")
    selected_taxonomy = taxonomy.loc[
        taxonomy["行业类型编码"].astype(str).eq("008002"), list(required)
    ].copy()
    codes = selected_taxonomy["类目编码"].astype(str)
    if codes.duplicated().any():
        raise CourageStrictC2IndustryR6Error("duplicate taxonomy code")
    parents = dict(
        zip(codes, selected_taxonomy["父类编码"].fillna("").astype(str), strict=True)
    )
    levels = dict(zip(codes, selected_taxonomy["分级"].astype(int), strict=True))
    names = dict(zip(codes, selected_taxonomy["类目名称"].astype(str), strict=True))

    def level2(code: str) -> tuple[str, str]:
        seen: set[str] = set()
        current = code
        while current and current not in seen:
            seen.add(current)
            if levels.get(current) == 2:
                return current, names[current]
            current = parents.get(current, "")
        raise CourageStrictC2IndustryR6Error(f"no level-2 ancestor: {code}")

    selected = raw.loc[raw["classification_standard_code"].eq("008002")].copy()
    if selected.duplicated(["instrument", "change_date"]).any():
        raise CourageStrictC2IndustryR6Error("duplicate PIT industry change key")
    sectors = [level2(code) for code in selected["industry_code"]]
    selected["sector_level2_code"] = [item[0] for item in sectors]
    selected["sector_level2_name"] = [item[1] for item in sectors]
    return (
        selected.loc[:, list(PIT_COLUMNS)]
        .sort_values(["instrument", "change_date"], kind="mergesort")
        .reset_index(drop=True)
    )


def assign_industry_strictly_before_v1(
    daily_keys: pd.DataFrame, pit_history: pd.DataFrame
) -> pd.DataFrame:
    required = {"instrument", "session_date"}
    if not required.issubset(daily_keys):
        raise CourageStrictC2IndustryR6Error("daily key schema drift")
    keys = daily_keys.loc[:, ["instrument", "session_date"]].copy()
    keys["session_date"] = pd.to_datetime(
        keys["session_date"], errors="raise"
    ).dt.normalize()
    if keys.duplicated(["instrument", "session_date"]).any():
        raise CourageStrictC2IndustryR6Error("duplicate daily industry key")
    histories = {
        symbol: rows.sort_values("change_date", kind="mergesort")
        for symbol, rows in pit_history.groupby("instrument", sort=False)
    }
    parts: list[pd.DataFrame] = []
    for symbol, target in keys.groupby("instrument", sort=False):
        source = histories.get(symbol)
        target = target.sort_values("session_date", kind="mergesort")
        if source is None:
            for column in PIT_COLUMNS[1:]:
                target[column] = pd.NA
        else:
            target = pd.merge_asof(
                target,
                source.loc[:, list(PIT_COLUMNS[1:])],
                left_on="session_date",
                right_on="change_date",
                direction="backward",
                allow_exact_matches=False,
            )
        parts.append(target)
    result = pd.concat(parts, ignore_index=True)
    known = result["sector_level2_code"].notna()
    result["industry_known"] = known
    result["same_change_date_use_authorized"] = False
    result["future_fill_authorized"] = False
    if result.loc[known, "change_date"].ge(result.loc[known, "session_date"]).any():
        raise CourageStrictC2IndustryR6Error("industry assignment is not strict PIT")
    return result.sort_values(
        ["session_date", "instrument"], kind="mergesort"
    ).reset_index(drop=True)


def _validate_controls(
    *, root: Path, config_path: Path, authorization_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path, "C2-R6 config")
    auth = _load_json(authorization_path, "C2-R6 authorization")
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("status") != "FROZEN_NOT_AUTHORIZED"
    ):
        raise CourageStrictC2IndustryR6Error("config identity drift")
    if config.get("record_read_end_exclusive") != "2026-04-01":
        raise CourageStrictC2IndustryR6Error("record boundary drift")
    if any(value is not False for value in config.get("authority_matrix", {}).values()):
        raise CourageStrictC2IndustryR6Error("config authorities must remain false")
    for label, identity in config["control_identities"].items():
        if sha256_file_v1(_resolve(root, identity["path"])) != identity["sha256"]:
            raise CourageStrictC2IndustryR6Error(f"control drift: {label}")
    if (
        auth.get("schema_version") != AUTH_SCHEMA
        or auth.get("operator_statement") != OPERATOR_STATEMENT
    ):
        raise CourageStrictC2IndustryR6Error("authorization identity drift")
    if (
        auth.get("operator_statement_sha256")
        != hashlib.sha256(OPERATOR_STATEMENT.encode()).hexdigest()
    ):
        raise CourageStrictC2IndustryR6Error("operator statement SHA drift")
    if (
        auth.get("one_time_authorization") is not True
        or auth.get("consumed_when_manifest_written") is not True
    ):
        raise CourageStrictC2IndustryR6Error("authorization lifecycle drift")
    if auth.get("config") != {
        "path": config_path.relative_to(root).as_posix(),
        "sha256": sha256_file_v1(config_path),
    }:
        raise CourageStrictC2IndustryR6Error("authorization config drift")
    runner = Path(__file__).resolve()
    if auth.get("runner") != {
        "path": runner.relative_to(root).as_posix(),
        "sha256": sha256_file_v1(runner),
    }:
        raise CourageStrictC2IndustryR6Error("authorization runner drift")
    authorities = auth.get("authorities")
    if (
        not isinstance(authorities, dict)
        or set(authorities) != TRUE_AUTHORITIES | FALSE_AUTHORITIES
    ):
        raise CourageStrictC2IndustryR6Error("authorization key drift")
    if any(authorities[key] is not True for key in TRUE_AUTHORITIES) or any(
        authorities[key] is not False for key in FALSE_AUTHORITIES
    ):
        raise CourageStrictC2IndustryR6Error("authorization scope drift")
    return config, auth


def run_courage_strict_c2_industry_r6_v1(
    *, project_root: Path, config_path: Path, authorization_path: Path
) -> Path:
    root = project_root.resolve()
    config_file, auth_file = config_path.resolve(), authorization_path.resolve()
    config, _ = _validate_controls(
        root=root, config_path=config_file, authorization_path=auth_file
    )
    config_sha, auth_sha = sha256_file_v1(config_file), sha256_file_v1(auth_file)
    target = root / config["output_root"] / f"authorization-{auth_sha[:24]}"
    manifest_path = target / "_industry_r6_manifest.json"
    if manifest_path.is_file():
        manifest = _load_json(manifest_path, "existing C2-R6 manifest")
        if (
            manifest.get("decision") != "PASS_C2_G10_INPUT_ADMISSION_READY"
            or manifest.get("config_sha256") != config_sha
            or manifest.get("authorization_sha256") != auth_sha
        ):
            raise CourageStrictC2IndustryR6Error("existing manifest drift")
        expected = {item["path"]: item["sha256"] for item in manifest["files"]}
        if set(expected) != {
            "raw_history.parquet",
            "pit_history.parquet",
            "daily_industry.parquet",
            "uncovered_instruments.parquet",
        }:
            raise CourageStrictC2IndustryR6Error("existing file set drift")
        for name, digest in expected.items():
            if sha256_file_v1(target / name) != digest:
                raise CourageStrictC2IndustryR6Error("existing file drift")
        return manifest_path

    inputs = {
        name: _resolve(root, identity["path"])
        for name, identity in config["inputs"].items()
    }
    for name, path in inputs.items():
        if sha256_file_v1(path) != config["inputs"][name]["sha256"]:
            raise CourageStrictC2IndustryR6Error(f"input drift: {name}")
    security = pq.read_table(
        inputs["security_master"],
        columns=["ts_code", "exchange", "list_date", "delist_date"],
    ).to_pandas(ignore_metadata=True)
    end = pd.Timestamp(config["record_read_end_exclusive"])
    start = pd.Timestamp(config["daily_assignment_start"])
    security["list_date"] = pd.to_datetime(security["list_date"], errors="raise")
    security["delist_date"] = pd.to_datetime(security["delist_date"], errors="coerce")
    admitted = sorted(
        security.loc[
            security["exchange"].isin(["SSE", "SZSE"])
            & security["list_date"].lt(end)
            & (security["delist_date"].isna() | security["delist_date"].ge(start)),
            "ts_code",
        ].astype(str)
    )
    if len(admitted) != config["expected_counts"]["admitted_instruments"]:
        raise CourageStrictC2IndustryR6Error("admitted scope drift")

    old_raw = pq.read_table(
        inputs["reusable_raw_history"], filters=[("change_date", "<", end)]
    ).to_pandas(ignore_metadata=True)
    old_raw = old_raw.rename(columns={"symbol": "instrument"}).loc[:, list(RAW_COLUMNS)]
    old_raw = old_raw.loc[old_raw["instrument"].isin(admitted)].copy()
    reusable_symbols = set(old_raw["instrument"].astype(str))
    missing_symbols = [symbol for symbol in admitted if symbol not in reusable_symbols]
    if (
        len(reusable_symbols) != config["expected_counts"]["reusable_symbols"]
        or len(missing_symbols) != config["expected_counts"]["symbols_to_fetch"]
    ):
        raise CourageStrictC2IndustryR6Error("reusable/fetch partition drift")

    scratch = root / config["scratch_root"] / f"authorization-{auth_sha[:24]}"
    if scratch.is_symlink():
        raise CourageStrictC2IndustryR6Error("unsafe scratch")
    cache = scratch / "raw-symbol"
    cache.mkdir(parents=True, exist_ok=True)
    import akshare as ak

    def fetch(symbol: str) -> Path:
        output = cache / f"{symbol}.parquet"
        if output.is_file():
            table = pq.read_table(output)
            if set(table.schema.names) != set(RAW_COLUMNS):
                raise CourageStrictC2IndustryR6Error("cached provider schema drift")
            return output
        last: BaseException | None = None
        frame: pd.DataFrame | None = None
        for attempt in range(int(config["provider"]["max_attempts"])):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    received = ak.stock_industry_change_cninfo(
                        symbol=symbol[:6],
                        start_date="19900101",
                        end_date="20260331",
                    )
                frame = normalize_provider_history_v1(symbol, received)
                break
            except KeyError as exc:
                if exc.args == ("变更日期",):
                    frame = pd.DataFrame(columns=RAW_COLUMNS)
                    break
                last = exc
            except (
                Exception
            ) as exc:  # provider/network failure is retried and then terminal
                last = exc
            if attempt + 1 < int(config["provider"]["max_attempts"]):
                time.sleep(min(0.5 * (2**attempt), 8.0))
        if frame is None:
            raise CourageStrictC2IndustryR6Error(
                f"provider fetch failed: {symbol}"
            ) from last
        temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
        pq.write_table(
            pa.Table.from_pandas(frame, preserve_index=False),
            temporary,
            compression="zstd",
        )
        os.replace(temporary, output)
        return output

    with futures.ThreadPoolExecutor(
        max_workers=int(config["provider"]["workers"])
    ) as executor:
        fetched_paths = list(executor.map(fetch, missing_symbols))
    fetched = pd.concat(
        [pq.read_table(path).to_pandas(ignore_metadata=True) for path in fetched_paths],
        ignore_index=True,
    )
    raw = (
        pd.concat([old_raw, fetched], ignore_index=True)
        .drop_duplicates()
        .sort_values(
            [
                "instrument",
                "change_date",
                "classification_standard_code",
                "industry_code",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    if raw["change_date"].ge(end).any() or not set(raw["instrument"]).issubset(
        admitted
    ):
        raise CourageStrictC2IndustryR6Error("raw history boundary drift")
    taxonomy = pq.read_table(inputs["taxonomy"]).to_pandas(ignore_metadata=True)
    pit = build_pit_history_v1(raw, taxonomy)

    daily = pq.read_table(
        inputs["normalized_daily"], columns=["instrument", "session_date"]
    ).to_pandas(ignore_metadata=True)
    assignment = assign_industry_strictly_before_v1(daily, pit)
    covered = int(assignment["industry_known"].sum())
    coverage = covered / len(assignment)
    per_date = assignment.groupby("session_date")["industry_known"].mean()
    uncovered = assignment.loc[
        ~assignment["industry_known"], ["instrument"]
    ].drop_duplicates()
    observed = {
        "admitted_instruments": len(admitted),
        "reusable_symbols": len(reusable_symbols),
        "fetched_symbols": len(missing_symbols),
        "symbols_with_selected_history": pit["instrument"].nunique(),
        "selected_history_rows": len(pit),
        "daily_rows": len(assignment),
        "covered_daily_rows": covered,
        "daily_coverage": coverage,
        "minimum_daily_coverage": float(per_date.min()),
        "uncovered_instruments": len(uncovered),
    }
    if (
        coverage < config["coverage_gate"]["minimum_row_coverage"]
        or float(per_date.min()) < config["coverage_gate"]["minimum_daily_coverage"]
    ):
        raise CourageStrictC2IndustryR6Error(
            f"industry coverage below gate: {observed}"
        )
    if (
        assignment["same_change_date_use_authorized"].any()
        or assignment["future_fill_authorized"].any()
    ):
        raise CourageStrictC2IndustryR6Error("industry PIT authority drift")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.{uuid.uuid4().hex}.staging"
    staging.mkdir(mode=0o755)
    try:
        output_frames = {
            "raw_history.parquet": raw,
            "pit_history.parquet": pit,
            "daily_industry.parquet": assignment,
            "uncovered_instruments.parquet": uncovered,
        }
        files = []
        for name, frame in output_frames.items():
            path = staging / name
            pq.write_table(
                pa.Table.from_pandas(frame, preserve_index=False),
                path,
                compression="zstd",
            )
            files.append(
                {
                    "path": name,
                    "sha256": sha256_file_v1(path),
                    "bytes": path.stat().st_size,
                    "rows": len(frame),
                }
            )
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "run_id": config["run_id"],
            "decision": "PASS_C2_G10_INPUT_ADMISSION_READY",
            "terminal_state": "STOP_AFTER_C2_G10_BEFORE_G11",
            "config_sha256": config_sha,
            "authorization_sha256": auth_sha,
            "authorization_consumed": True,
            "provider": config["provider"],
            "pit_contract": config["pit_contract"],
            "coverage_gate": config["coverage_gate"],
            "coverage": observed,
            "files": files,
            "development_test_read": False,
            "reserved_confirm_read": False,
            "holdout_read": False,
            "training_executed": False,
            "remote_push_executed": False,
        }
        _atomic_json(staging / manifest_path.name, manifest)
        os.replace(staging, target)
        shutil.rmtree(scratch, ignore_errors=True)
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
    print(
        run_courage_strict_c2_industry_r6_v1(
            project_root=args.project_root,
            config_path=args.config,
            authorization_path=args.authorization,
        )
    )


if __name__ == "__main__":
    main()

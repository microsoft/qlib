"""C2-R3 PIT reconstruction for corporate actions and adjustment factors.

The legacy verifier retained the ex-date and record date but discarded CNInfo's
implementation-plan announcement date.  This bounded remediation reconstructs
all adjustment-factor changes for the C1 Shanghai/Shenzhen A-share scope and
joins them to the official CNInfo dividend interface.  Date-only announcement
metadata is converted to a conservative end-of-calendar-day upper bound; it is
never treated as intraday-ready on the announcement date.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import inspect
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Final

import akshare as ak
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from qlib.contrib.data.courage_strict_v1.minute_courage_strict_c2_identity_v1 import (
    sha256_file_v1,
)


class CourageStrictC2CorporateActionR3Error(RuntimeError):
    """Raised when the bounded G07 reconstruction cannot be proven."""


CONFIG_SCHEMA: Final[str] = "courage_strict_c2_corporate_action_r3_config_v1"
AUTH_SCHEMA: Final[str] = "courage_strict_c2_corporate_action_r3_authorization_v1"
MANIFEST_SCHEMA: Final[str] = "courage_strict_c2_corporate_action_r3_manifest_v1"
OPERATOR_STATEMENT: Final[str] = (
    "请把下面的目标提交到著目录空间下的持久Codex.sh。确认任务已经启动并给出任务编号后再回复我。"
    "我提供给你所有权限，你需要充分利用资源，并且以较快的速度完成C0-C8任务。"
)
TRUE_AUTHORITIES: Final[set[str]] = {
    "execution_authorized",
    "official_corporate_action_metadata_read_authorized",
    "source_record_read_authorized",
    "c2_corporate_action_pit_reconstruction_authorized",
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
CANONICAL_COLUMNS: Final[tuple[str, ...]] = (
    "instrument",
    "action_type",
    "announcement_time",
    "announcement_date",
    "announcement_time_resolution",
    "same_announcement_day_intraday_use_authorized",
    "record_date",
    "ex_date",
    "effective_time",
    "adjustment_factor",
    "previous_adjustment_factor",
    "factor_ratio",
    "cash_dividend_per_10",
    "stock_dividend_per_10",
    "capitalization_per_10",
    "implementation_description",
    "report_period",
    "source",
    "source_version",
)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CourageStrictC2CorporateActionR3Error(f"unsafe or missing {label}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CourageStrictC2CorporateActionR3Error(f"invalid {label}") from exc
    if not isinstance(value, dict):
        raise CourageStrictC2CorporateActionR3Error(f"{label} must be an object")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CourageStrictC2CorporateActionR3Error(
            "configured path escapes project"
        ) from exc
    return path


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_bytes_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_controls(
    *, root: Path, config_path: Path, authorization_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path, "C2-R3 corporate-action config")
    authorization = _load_json(
        authorization_path, "C2-R3 corporate-action authorization"
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise CourageStrictC2CorporateActionR3Error("config schema drift")
    if config.get("status") != "FROZEN_NOT_AUTHORIZED":
        raise CourageStrictC2CorporateActionR3Error("config status drift")
    if config.get("record_read_interval") != ["2025-04-01", "2026-04-01"]:
        raise CourageStrictC2CorporateActionR3Error("record interval drift")
    if any(value is not False for value in config.get("authority_matrix", {}).values()):
        raise CourageStrictC2CorporateActionR3Error(
            "config authorities must remain false"
        )
    for label, identity in config["control_identities"].items():
        if sha256_file_v1(_resolve(root, identity["path"])) != identity["sha256"]:
            raise CourageStrictC2CorporateActionR3Error(
                f"control identity drift: {label}"
            )
    for label, identity in config["source_inputs"].items():
        path = Path(identity["absolute_path"])
        if path.is_symlink() or not path.is_file():
            raise CourageStrictC2CorporateActionR3Error(f"unsafe source: {label}")
        if sha256_file_v1(path) != identity["sha256"]:
            raise CourageStrictC2CorporateActionR3Error(
                f"source identity drift: {label}"
            )
    if authorization.get("schema_version") != AUTH_SCHEMA:
        raise CourageStrictC2CorporateActionR3Error("authorization schema drift")
    if authorization.get("one_time_authorization") is not True:
        raise CourageStrictC2CorporateActionR3Error("authorization must be one-time")
    if authorization.get("consumed_when_manifest_written") is not True:
        raise CourageStrictC2CorporateActionR3Error("consumption rule drift")
    if authorization.get("operator_statement") != OPERATOR_STATEMENT:
        raise CourageStrictC2CorporateActionR3Error("operator statement drift")
    statement_sha = hashlib.sha256(OPERATOR_STATEMENT.encode()).hexdigest()
    if authorization.get("operator_statement_sha256") != statement_sha:
        raise CourageStrictC2CorporateActionR3Error("operator statement SHA drift")
    if authorization.get("config") != {
        "path": config_path.relative_to(root).as_posix(),
        "sha256": sha256_file_v1(config_path),
    }:
        raise CourageStrictC2CorporateActionR3Error(
            "authorization config identity drift"
        )
    runner = Path(__file__).resolve()
    if authorization.get("runner") != {
        "path": runner.relative_to(root).as_posix(),
        "sha256": sha256_file_v1(runner),
    }:
        raise CourageStrictC2CorporateActionR3Error(
            "authorization runner identity drift"
        )
    authorities = authorization.get("authorities")
    if not isinstance(authorities, dict) or set(authorities) != (
        TRUE_AUTHORITIES | FALSE_AUTHORITIES
    ):
        raise CourageStrictC2CorporateActionR3Error("authorization key drift")
    if any(authorities[key] is not True for key in TRUE_AUTHORITIES):
        raise CourageStrictC2CorporateActionR3Error("required authority is false")
    if any(authorities[key] is not False for key in FALSE_AUTHORITIES):
        raise CourageStrictC2CorporateActionR3Error("forbidden authority is true")
    return config, authorization


def derive_factor_changes_v1(
    factors: pd.DataFrame,
    *,
    admitted_instruments: set[str],
    interval: tuple[str, str],
) -> pd.DataFrame:
    """Derive every in-scope cumulative-factor change without old PIT membership."""

    required = {"symbol", "trade_date", "adj_factor", "event_time"}
    missing = sorted(required - set(factors))
    if missing:
        raise CourageStrictC2CorporateActionR3Error(f"factor fields missing: {missing}")
    frame = factors.loc[
        factors["symbol"].isin(admitted_instruments), list(required)
    ].copy()
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"], errors="raise"
    ).dt.normalize()
    frame["adj_factor"] = pd.to_numeric(frame["adj_factor"], errors="raise")
    if frame.duplicated(["symbol", "trade_date"]).any():
        raise CourageStrictC2CorporateActionR3Error("duplicate factor key")
    frame = frame.sort_values(["symbol", "trade_date"], kind="mergesort")
    frame["previous_adjustment_factor"] = frame.groupby("symbol", sort=False)[
        "adj_factor"
    ].shift()
    start, end = map(pd.Timestamp, interval)
    changed = frame.loc[
        frame["trade_date"].ge(start)
        & frame["trade_date"].lt(end)
        & frame["previous_adjustment_factor"].notna()
        & ~frame["adj_factor"].eq(frame["previous_adjustment_factor"])
    ].copy()
    if changed.empty:
        raise CourageStrictC2CorporateActionR3Error("no factor changes in scope")
    if (~changed["adj_factor"].gt(0)).any() or (
        ~changed["previous_adjustment_factor"].gt(0)
    ).any():
        raise CourageStrictC2CorporateActionR3Error("non-positive factor")
    changed["factor_ratio"] = (
        changed["adj_factor"] / changed["previous_adjustment_factor"]
    )
    return changed.reset_index(drop=True)


def normalize_cninfo_dividend_v1(raw: pd.DataFrame, *, instrument: str) -> pd.DataFrame:
    """Normalize the complete provider table while retaining announcement date."""

    required = {
        "实施方案公告日期",
        "分红类型",
        "送股比例",
        "转增比例",
        "派息比例",
        "股权登记日",
        "除权日",
        "实施方案分红说明",
        "报告时间",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise CourageStrictC2CorporateActionR3Error(
            f"CNInfo dividend schema changed for {instrument}: {missing}"
        )
    frame = raw.copy()
    frame["instrument"] = instrument
    date_mapping = {
        "实施方案公告日期": "announcement_date",
        "股权登记日": "record_date",
        "除权日": "ex_date",
    }
    for source, target in date_mapping.items():
        frame[target] = pd.to_datetime(frame[source], errors="coerce").dt.normalize()
    numeric_mapping = {
        "派息比例": "cash_dividend_per_10",
        "送股比例": "stock_dividend_per_10",
        "转增比例": "capitalization_per_10",
    }
    for source, target in numeric_mapping.items():
        frame[target] = pd.to_numeric(frame[source], errors="coerce").fillna(0.0)
    frame["action_type"] = frame["分红类型"].astype("string")
    frame["implementation_description"] = frame["实施方案分红说明"].astype("string")
    frame["report_period"] = frame["报告时间"].astype("string")
    has_action = frame[list(numeric_mapping.values())].abs().gt(0).any(axis=1)
    frame = frame.loc[has_action & frame["ex_date"].notna()].copy()
    columns = [
        "instrument",
        "action_type",
        "announcement_date",
        "record_date",
        "ex_date",
        *numeric_mapping.values(),
        "implementation_description",
        "report_period",
    ]
    return frame[columns].reset_index(drop=True)


def build_canonical_actions_v1(
    *, changes: pd.DataFrame, announcements: pd.DataFrame, official_axis: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join factor changes to announcements and return canonical rows + unmatched."""

    first_slot = official_axis.loc[
        official_axis["minute_slot"].eq(0), ["session_date", "minute_window_start"]
    ].copy()
    first_slot["session_date"] = pd.to_datetime(
        first_slot["session_date"]
    ).dt.normalize()
    if first_slot.groupby("session_date")["minute_window_start"].nunique().ne(1).any():
        raise CourageStrictC2CorporateActionR3Error("exchange session-open mismatch")
    first_slot = first_slot.drop_duplicates(["session_date"], keep="first")
    candidate = changes.rename(columns={"symbol": "instrument"}).merge(
        announcements,
        left_on=["instrument", "trade_date"],
        right_on=["instrument", "ex_date"],
        how="left",
        validate="one_to_many",
        indicator=True,
    )
    matched = candidate.loc[candidate["_merge"].eq("both")].copy()
    if not matched.empty:
        record_equal = matched["record_date"].isna() | matched["record_date"].eq(
            matched["trade_date"] - pd.Timedelta(days=1)
        )
        # Prefer an exact record-date match when one is available, otherwise retain
        # the ex-date match because holidays make calendar-day subtraction unsafe.
        matched["record_match_priority"] = record_equal.astype("int8")
        matched = matched.sort_values(
            ["instrument", "trade_date", "record_match_priority", "announcement_date"],
            ascending=[True, True, False, False],
            kind="mergesort",
        ).drop_duplicates(["instrument", "trade_date"], keep="first")
    matched_keys = pd.MultiIndex.from_frame(matched[["instrument", "trade_date"]])
    change_keys = pd.MultiIndex.from_frame(
        changes.rename(columns={"symbol": "instrument"})[["instrument", "trade_date"]]
    )
    unmatched = (
        changes.loc[~change_keys.isin(matched_keys)].copy().reset_index(drop=True)
    )
    if matched.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS), unmatched
    matched = matched.merge(
        first_slot,
        left_on="trade_date",
        right_on="session_date",
        how="left",
        validate="many_to_one",
    )
    if matched["minute_window_start"].isna().any():
        raise CourageStrictC2CorporateActionR3Error("ex-date absent from official axis")
    announcement_date = pd.to_datetime(matched["announcement_date"], errors="coerce")
    matched["announcement_time"] = (
        announcement_date.dt.tz_localize("Asia/Shanghai")
        + pd.Timedelta(days=1)
        - pd.Timedelta(nanoseconds=1)
    )
    matched["announcement_time_resolution"] = "calendar_date_conservative_end_of_day"
    matched["same_announcement_day_intraday_use_authorized"] = False
    matched["effective_time"] = matched["minute_window_start"]
    matched["adjustment_factor"] = matched["adj_factor"]
    matched["source"] = "CNInfo_via_AKShare_stock_dividend_cninfo"
    matched["source_version"] = importlib.metadata.version("akshare")
    if matched["announcement_time"].isna().any():
        raise CourageStrictC2CorporateActionR3Error("missing announcement date")
    if ~(matched["announcement_time"] < matched["effective_time"]).all():
        raise CourageStrictC2CorporateActionR3Error(
            "announcement upper bound is not before action effective time"
        )
    result = matched
    result = result.loc[:, list(CANONICAL_COLUMNS)].sort_values(
        ["instrument", "ex_date"], kind="mergesort"
    )
    if result.duplicated(["instrument", "ex_date"]).any():
        raise CourageStrictC2CorporateActionR3Error(
            "duplicate canonical corporate action"
        )
    return result.reset_index(drop=True), unmatched


def _fetch_one(
    symbol: str, max_attempts: int
) -> tuple[str, pd.DataFrame | None, str | None]:
    code = symbol.split(".")[0]
    last: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return symbol, ak.stock_dividend_cninfo(symbol=code), None
        except KeyError as exc:
            # AKShare constructs an empty DataFrame when CNInfo returns no records,
            # then indexes the expected columns and raises this exact KeyError.
            # Preserve the official empty result; do not misclassify it as a
            # transient transport failure.
            if exc.args == ("实施方案公告日期",):
                columns = [
                    "实施方案公告日期",
                    "分红类型",
                    "送股比例",
                    "转增比例",
                    "派息比例",
                    "股权登记日",
                    "除权日",
                    "派息日",
                    "股份到账日",
                    "实施方案分红说明",
                    "报告时间",
                ]
                return symbol, pd.DataFrame(columns=columns), None
            last = exc
            if attempt < max_attempts:
                time.sleep(float(attempt))
        except Exception as exc:  # pragma: no cover - remote provider failures vary
            last = exc
            if attempt < max_attempts:
                time.sleep(float(attempt))
    return symbol, None, repr(last)


def fetch_announcements_v1(
    symbols: list[str], *, workers: int, max_attempts: int
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    if workers <= 0 or max_attempts <= 0:
        raise CourageStrictC2CorporateActionR3Error("invalid fetch concurrency")
    frames: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(workers, len(symbols))
    ) as pool:
        for index, (symbol, raw, error) in enumerate(
            pool.map(lambda value: _fetch_one(value, max_attempts), symbols), start=1
        ):
            if error is not None or raw is None:
                errors.append({"instrument": symbol, "error": error or "missing frame"})
            else:
                frames.append(normalize_cninfo_dividend_v1(raw, instrument=symbol))
            if index % 250 == 0:
                print(
                    f"CNInfo corporate actions {index}/{len(symbols)}; errors={len(errors)}",
                    flush=True,
                )
    if errors:
        retry_frames: list[pd.DataFrame] = []
        final_errors: list[dict[str, str]] = []
        for item in errors:
            symbol = item["instrument"]
            _, raw, error = _fetch_one(symbol, max_attempts * 2)
            if error is not None or raw is None:
                final_errors.append(
                    {"instrument": symbol, "error": error or "missing frame"}
                )
            else:
                retry_frames.append(
                    normalize_cninfo_dividend_v1(raw, instrument=symbol)
                )
        frames.extend(retry_frames)
        errors = final_errors
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return result, errors


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, path, compression="zstd", use_dictionary=True)


def _verify_existing(
    manifest_path: Path, *, config_sha: str, authorization_sha: str
) -> Path | None:
    if not manifest_path.is_file():
        return None
    manifest = _load_json(manifest_path, "existing C2-R3 corporate-action manifest")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("decision") != "PASS_C2_G07_CORPORATE_ACTION_PIT_RECONSTRUCTION"
        or manifest.get("config_sha256") != config_sha
        or manifest.get("authorization_sha256") != authorization_sha
        or manifest.get("authorization_consumed") is not True
        or manifest.get("all_downstream_authorities_false") is not True
    ):
        raise CourageStrictC2CorporateActionR3Error("existing manifest drift")
    for item in manifest.get("files", []):
        path = manifest_path.parent / item["path"]
        if path.is_symlink() or sha256_file_v1(path) != item["sha256"]:
            raise CourageStrictC2CorporateActionR3Error("existing artifact drift")
    return manifest_path


def run_courage_strict_c2_corporate_action_r3_v1(
    *, project_root: Path, config_path: Path, authorization_path: Path
) -> Path:
    root = project_root.resolve()
    config_file = config_path.resolve()
    authorization_file = authorization_path.resolve()
    config, authorization = _validate_controls(
        root=root, config_path=config_file, authorization_path=authorization_file
    )
    config_sha = sha256_file_v1(config_file)
    auth_sha = sha256_file_v1(authorization_file)
    target = _resolve(root, config["output_root"]) / f"authorization-{auth_sha[:24]}"
    manifest_path = target / "_corporate_action_r3_manifest.json"
    existing = _verify_existing(
        manifest_path, config_sha=config_sha, authorization_sha=auth_sha
    )
    if existing is not None:
        return existing
    if target.exists():
        raise CourageStrictC2CorporateActionR3Error("incomplete target exists")
    staging = target.with_name(f".{target.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    try:
        security = pd.read_parquet(
            Path(config["source_inputs"]["security_master"]["absolute_path"]),
            columns=[
                "ts_code",
                "exchange",
                "market",
                "curr_type",
                "list_date",
                "delist_date",
            ],
        )
        start, end = config["record_read_interval"]
        admitted = security.loc[
            security["exchange"].isin(["SSE", "SZSE"])
            & security["curr_type"].eq("CNY")
            & security["market"].isin(["主板", "创业板", "科创板"])
            & security["ts_code"].str.endswith((".SH", ".SZ"))
            & security["list_date"].lt(pd.Timestamp(end))
            & (
                security["delist_date"].isna()
                | security["delist_date"].ge(pd.Timestamp(start))
            ),
            "ts_code",
        ]
        admitted_set = set(admitted.astype(str))
        factors = pd.concat(
            [
                pd.read_parquet(
                    Path(config["source_inputs"][key]["absolute_path"]),
                    columns=["symbol", "trade_date", "event_time", "adj_factor"],
                )
                for key in ("adjustment_factor_2025", "adjustment_factor_2026")
            ],
            ignore_index=True,
        )
        changes = derive_factor_changes_v1(
            factors,
            admitted_instruments=admitted_set,
            interval=(start, end),
        )
        symbols = sorted(changes["symbol"].unique())
        announcements, errors = fetch_announcements_v1(
            symbols,
            workers=int(config["fetch_policy"]["workers"]),
            max_attempts=int(config["fetch_policy"]["max_attempts"]),
        )
        if errors:
            _write_bytes_exclusive(
                staging / "fetch_errors.json", _canonical_json_bytes(errors)
            )
            raise CourageStrictC2CorporateActionR3Error(
                f"CNInfo fetch incomplete for {len(errors)} instruments"
            )
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        announcements = announcements.loc[
            announcements["ex_date"].ge(start_ts) & announcements["ex_date"].lt(end_ts)
        ].copy()
        axis = pd.read_parquet(
            Path(config["source_inputs"]["official_axis"]["absolute_path"]),
            columns=["session_date", "minute_slot", "minute_window_start"],
        )
        canonical, unmatched = build_canonical_actions_v1(
            changes=changes, announcements=announcements, official_axis=axis
        )
        snapshot_path = staging / "cninfo_dividend_snapshot.parquet"
        changes_path = staging / "factor_changes.parquet"
        actions_path = staging / "effective_dated_corporate_actions.parquet"
        unmatched_path = staging / "unmatched_factor_changes.parquet"
        _write_parquet(snapshot_path, announcements)
        _write_parquet(changes_path, changes)
        _write_parquet(actions_path, canonical)
        _write_parquet(unmatched_path, unmatched)
        expected = config["expected_coverage"]
        observed = {
            "admitted_instruments": len(admitted_set),
            "factor_change_rows": len(changes),
            "factor_change_instruments": int(changes["symbol"].nunique()),
            "matched_action_rows": len(canonical),
            "unmatched_factor_change_rows": len(unmatched),
        }
        # All factor changes must be explained; partial provider coverage is a
        # terminal G07 failure and cannot be silently accepted.
        decision = (
            "PASS_C2_G07_CORPORATE_ACTION_PIT_RECONSTRUCTION"
            if unmatched.empty
            else "FAIL_C2_G07_UNMATCHED_FACTOR_CHANGES"
        )
        if expected and any(
            observed.get(key) != value for key, value in expected.items()
        ):
            decision = "FAIL_C2_G07_COVERAGE_DRIFT"
        files = []
        for path in (snapshot_path, changes_path, actions_path, unmatched_path):
            files.append(
                {
                    "path": path.name,
                    "sha256": sha256_file_v1(path),
                    "bytes": path.stat().st_size,
                    "rows": pq.ParquetFile(path).metadata.num_rows,
                }
            )
        ak_module_path = Path(inspect.getfile(ak.stock_dividend_cninfo)).resolve()
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "decision": decision,
            "terminal_state": "STOP_AFTER_G07_REMEDIATION_BEFORE_REAUDIT",
            "config_sha256": config_sha,
            "authorization_sha256": auth_sha,
            "authorization_consumed": True,
            "operator_statement_sha256": authorization["operator_statement_sha256"],
            "record_read_interval": config["record_read_interval"],
            "coverage": observed,
            "canonical_columns": list(CANONICAL_COLUMNS),
            "announcement_time_semantics": {
                "source_field": "实施方案公告日期",
                "source_resolution": "calendar_date",
                "canonical_upper_bound": "23:59:59.999999999_Asia/Shanghai",
                "same_announcement_day_intraday_use_authorized": False,
                "required_relation": "announcement_time < effective_time",
            },
            "factor_change_policy": (
                "all_cumulative_factor_changes_in_C1_SH_SZ_scope_without_old_PIT_membership"
            ),
            "official_source": {
                "owner": "CNInfo",
                "endpoint": "https://webapi.cninfo.com.cn/api/sysapi/p_sysapi1139",
                "provider_function": "akshare.stock_dividend_cninfo",
                "provider_version": importlib.metadata.version("akshare"),
                "provider_module_path": str(ak_module_path),
                "provider_module_sha256": sha256_file_v1(ak_module_path),
            },
            "files": files,
            "development_test_read": False,
            "reserved_confirm_read": False,
            "holdout_read": False,
            "universe_or_dataset_materialized": False,
            "feature_label_sequence_executed": False,
            "runtime_training_selection_executed": False,
            "refit_executed": False,
            "strategy_backtest_trading_executed": False,
            "remote_push_executed": False,
            "all_downstream_authorities_false": True,
            "terminal_authority_matrix": {
                key: False for key in sorted(TRUE_AUTHORITIES | FALSE_AUTHORITIES)
            },
        }
        _write_bytes_exclusive(
            staging / "_corporate_action_r3_manifest.json",
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
        run_courage_strict_c2_corporate_action_r3_v1(
            project_root=args.project_root,
            config_path=args.config,
            authorization_path=args.authorization,
        )
    )


if __name__ == "__main__":
    main()

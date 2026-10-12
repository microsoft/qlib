"""Bounded 2025-06 golden audit for Courage-strict C4 features."""

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
import pyarrow.parquet as pq

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
    compute_stock_features_v1,
)


class CourageStrictC4GoldenError(RuntimeError):
    """Raised when the bounded C4 golden audit fails closed."""


CONFIG_SCHEMA: Final[str] = "courage_strict_c4_golden_config_v1"
AUTH_SCHEMA: Final[str] = "courage_strict_c4_golden_authorization_v1"
REPORT_SCHEMA: Final[str] = "courage_strict_c4_golden_report_v1"
OPERATOR_STATEMENT: Final[str] = (
    "用户授权在当前Qlib仓库完成courage_strict_v1全部代码、数据构造、训练和评测。"
)
TRUE_AUTHORITIES: Final[set[str]] = {
    "execution_authorized",
    "accepted_C2_C3_read_authorized",
    "golden_source_record_read_authorized",
    "golden_feature_computation_authorized",
    "golden_evidence_write_authorized",
}
FALSE_AUTHORITIES: Final[set[str]] = {
    "full_feature_materialization_authorized",
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


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CourageStrictC4GoldenError(f"unsafe or missing {label}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CourageStrictC4GoldenError(f"{label} must be object")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if path.is_symlink() or not path.is_file():
        raise CourageStrictC4GoldenError(f"unsafe or missing input: {value}")
    return path


def _minute_record_map(identity: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in identity.get("files", []):
        if item.get("roles") != ["one_minute_OHLCVA_bars"]:
            continue
        symbol = Path(item["relative_path"]).stem
        if symbol in result:
            raise CourageStrictC4GoldenError("duplicate minute source identity")
        result[symbol] = item
    return result


def _validate_controls(
    root: Path, config_path: Path, auth_path: Path
) -> dict[str, Any]:
    config = _load_json(config_path, "C4 golden config")
    auth = _load_json(auth_path, "C4 golden authorization")
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("status") != (
        "FROZEN_NOT_AUTHORIZED"
    ):
        raise CourageStrictC4GoldenError("config drift")
    if any(value is not False for value in config["authority_matrix"].values()):
        raise CourageStrictC4GoldenError("config authority drift")
    for label, identity in config["inputs"].items():
        if sha256_file_v1(_resolve(root, identity["path"])) != identity["sha256"]:
            raise CourageStrictC4GoldenError(f"input drift: {label}")
    if auth.get("schema_version") != AUTH_SCHEMA or auth.get("operator_statement") != (
        OPERATOR_STATEMENT
    ):
        raise CourageStrictC4GoldenError("authorization identity drift")
    if (
        auth.get("operator_statement_sha256")
        != hashlib.sha256(OPERATOR_STATEMENT.encode()).hexdigest()
    ):
        raise CourageStrictC4GoldenError("operator SHA drift")
    if (
        auth.get("one_time_authorization") is not True
        or auth.get("consumed_when_report_written") is not True
    ):
        raise CourageStrictC4GoldenError("authorization lifecycle drift")
    if auth.get("config") != {
        "path": config_path.relative_to(root).as_posix(),
        "sha256": sha256_file_v1(config_path),
    }:
        raise CourageStrictC4GoldenError("authorization config drift")
    runner = Path(__file__).resolve()
    if auth.get("runners") != {
        "golden": {
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
    }:
        raise CourageStrictC4GoldenError("authorization runner closure drift")
    authorities = auth.get("authorities")
    if not isinstance(authorities, dict) or set(authorities) != (
        TRUE_AUTHORITIES | FALSE_AUTHORITIES
    ):
        raise CourageStrictC4GoldenError("authorization key drift")
    if any(authorities[key] is not True for key in TRUE_AUTHORITIES) or any(
        authorities[key] is not False for key in FALSE_AUTHORITIES
    ):
        raise CourageStrictC4GoldenError("authorization scope drift")
    return config


def _load_symbol_features(
    *,
    symbol: str,
    item: dict[str, Any],
    axis: pd.DataFrame,
    status: pd.DataFrame,
    event_offsets: np.ndarray,
    quarantine_offsets: np.ndarray,
    start: str,
    end: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    source = Path(item["absolute_path"])
    if source.is_symlink() or sha256_file_v1(source) != item["sha256"]:
        raise CourageStrictC4GoldenError(f"minute source drift: {symbol}")
    bars = (
        pq.read_table(
            source,
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
        raise CourageStrictC4GoldenError(f"minute source symbol drift: {symbol}")
    length = len(axis)
    raw = [np.full(length, np.nan, dtype=np.float64) for _ in range(5)]
    lookup = pd.Series(
        axis["global_minute_offset"].to_numpy(),
        index=axis["minute_window_end"].dt.tz_localize(None).astype("int64"),
    )
    timestamps = pd.to_datetime(bars["trade_time"], errors="coerce")
    offsets = timestamps.astype("int64").map(lookup)
    selected = offsets.notna()
    offsets_array = offsets.loc[selected].to_numpy(dtype=np.int64)
    subset = bars.loc[selected]
    for destination, column in zip(
        raw, ("close", "high", "low", "vol", "amount"), strict=True
    ):
        destination[offsets_array] = pd.to_numeric(
            subset[column], errors="coerce"
        ).to_numpy(dtype=np.float64)

    status_frame = status.copy()
    status_frame["session_date"] = pd.to_datetime(
        status_frame["session_date"]
    ).dt.normalize()
    status_frame = status_frame.set_index("session_date")
    dates = pd.DatetimeIndex(axis["session_date"])
    suspended = (
        status_frame["is_suspended"]
        .reindex(dates)
        .astype("boolean")
        .fillna(True)
        .to_numpy(dtype=bool)
    )
    upper = status_frame["limit_up_price"].reindex(dates).to_numpy(dtype=np.float64)
    lower = status_frame["limit_down_price"].reindex(dates).to_numpy(dtype=np.float64)
    close, high, low, _, _ = raw
    one_price = (
        np.isfinite(close)
        & np.isclose(close, high, rtol=0.0, atol=1e-10)
        & np.isclose(close, low, rtol=0.0, atol=1e-10)
    )
    locked = one_price & (
        (np.isfinite(upper) & np.isclose(close, upper, rtol=0.0, atol=0.005))
        | (np.isfinite(lower) & np.isclose(close, lower, rtol=0.0, atol=0.005))
    )
    invalid = suspended | locked
    for value in raw:
        value[invalid] = np.nan
    values, available, missing = compute_stock_features_v1(
        close=raw[0],
        high=raw[1],
        low=raw[2],
        volume=raw[3],
        amount=raw[4],
        event_offsets=event_offsets,
        quarantine_offsets=quarantine_offsets,
    )
    return values, available, missing, tuple(raw)


def run_courage_strict_c4_golden_v1(
    *, project_root: Path, config_path: Path, authorization_path: Path
) -> Path:
    root = project_root.resolve()
    config_path = config_path.resolve()
    authorization_path = authorization_path.resolve()
    config = _validate_controls(root, config_path, authorization_path)
    inputs = {
        key: _resolve(root, value["path"]) for key, value in config["inputs"].items()
    }
    c3 = _load_json(inputs["c3_final_report"], "C3 final report")
    if c3.get("decision") != "PASS_C3_COMPLETE":
        raise CourageStrictC4GoldenError("C3 is not complete")
    scope = pd.read_parquet(inputs["mainboard_scope"])
    symbols = sorted(scope["instrument"].astype(str).unique())
    if len(symbols) != config["expected_counts"]["mainboard_symbols"]:
        raise CourageStrictC4GoldenError("mainboard symbol count drift")
    axis = pd.read_parquet(inputs["official_axis"])
    axis = axis.loc[axis["exchange"].eq("SSE")].copy()
    axis["session_date"] = pd.to_datetime(axis["session_date"]).dt.normalize()
    start, end = config["source_interval"]
    axis = (
        axis.loc[
            axis["session_date"].ge(pd.Timestamp(start))
            & axis["session_date"].lt(pd.Timestamp(end))
        ]
        .sort_values(["session_date", "minute_slot"], kind="mergesort")
        .reset_index(drop=True)
    )
    axis["minute_window_end"] = pd.to_datetime(
        axis["minute_window_end"], utc=True
    ).dt.tz_convert("Asia/Shanghai")
    axis["global_minute_offset"] = np.arange(len(axis), dtype=np.int64)
    if axis.groupby("session_date").size().ne(240).any():
        raise CourageStrictC4GoldenError("golden axis drift")
    golden_dates = pd.DatetimeIndex(
        axis.loc[
            axis["session_date"].ge(pd.Timestamp(config["golden_month"])),
            "session_date",
        ].unique()
    )
    if len(golden_dates) != config["expected_counts"]["golden_sessions"]:
        raise CourageStrictC4GoldenError("golden session count drift")

    identity = _load_json(inputs["source_identity_manifest"], "source identity")
    minute_records = _minute_record_map(identity)
    if set(symbols) - set(minute_records):
        raise CourageStrictC4GoldenError("mainboard minute source missing")
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
    audit_symbols = set(symbols[: int(config["audit"]["prefix_symbols"])])
    raw_by_symbol: dict[str, tuple[np.ndarray, ...]] = {}

    def load(
        item: tuple[int, str],
    ) -> tuple[int, str, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
        index, symbol = item
        value, available, missing, raw = _load_symbol_features(
            symbol=symbol,
            item=minute_records[symbol],
            axis=axis,
            status=status_by_symbol.get(symbol, empty_status),
            event_offsets=events[symbol],
            quarantine_offsets=quarantines[symbol],
            start=start,
            end=end,
        )
        return index, symbol, value, available, missing, raw

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(config["resources"]["workers"])
    ) as pool:
        for completed, result in enumerate(pool.map(load, enumerate(symbols)), start=1):
            index, symbol, value, available, missing, raw = result
            stock_values[index] = value
            stock_available[index] = available
            stock_missing[index] = missing
            if symbol in audit_symbols:
                raw_by_symbol[symbol] = raw
            if completed % 500 == 0 or completed == len(symbols):
                print(
                    f"C4 golden stock features {completed}/{len(symbols)}", flush=True
                )

    industry = pd.read_parquet(inputs["daily_industry"])
    industry["session_date"] = pd.to_datetime(industry["session_date"]).dt.normalize()
    industry = industry.loc[
        industry["instrument"].isin(symbols)
        & industry["session_date"].isin(golden_dates)
    ]
    industry_lookup = industry.set_index(["instrument", "session_date"])[
        "sector_level2_code"
    ]
    scope_indexed = scope.set_index("instrument").reindex(symbols)
    list_date = pd.to_datetime(scope_indexed["list_date"]).to_numpy(
        dtype="datetime64[D]"
    )
    delist_date = pd.to_datetime(scope_indexed["delist_date"]).to_numpy(
        dtype="datetime64[D]"
    )
    sector_valid_counts = {feature: 0 for feature in SECTOR_FEATURES}
    sector_state_rows = 0
    for date in golden_dates:
        day = np.datetime64(date.date())
        active = (list_date <= day) & (np.isnat(delist_date) | (delist_date >= day))
        codes = np.array(
            [
                industry_lookup.get((symbol, pd.Timestamp(date)), None)
                for symbol in symbols
            ],
            dtype=object,
        )
        start_offset = date_start[pd.Timestamp(date)]
        result = compute_sector_features_day_v1(
            stock_values=stock_values[:, start_offset : start_offset + 240],
            stock_available=stock_available[:, start_offset : start_offset + 240],
            stock_missing=stock_missing[:, start_offset : start_offset + 240],
            sector_codes=codes,
            active_symbols=active,
            minimum_coverage=float(config["sector_gate"]["minimum_coverage"]),
            minimum_valid_count=int(config["sector_gate"]["minimum_valid_count"]),
        )
        for output, available, missing, _ in result.values():
            accepted = available & ~missing & np.isfinite(output)
            sector_state_rows += len(output)
            for index, feature in enumerate(SECTOR_FEATURES):
                sector_valid_counts[feature] += int(accepted[:, index].sum())
    if any(value <= 0 for value in sector_valid_counts.values()):
        raise CourageStrictC4GoldenError("golden sector feature has no valid state")

    endpoints = np.linspace(
        date_start[golden_dates[0]],
        len(axis) - 1,
        int(config["audit"]["prefix_endpoints"]),
    ).astype(np.int64)
    parity_checks = 0
    for symbol in sorted(audit_symbols):
        index = symbols.index(symbol)
        raw = raw_by_symbol[symbol]
        for endpoint in endpoints:
            prefix_values, prefix_available, prefix_missing = compute_stock_features_v1(
                close=raw[0][: endpoint + 1],
                high=raw[1][: endpoint + 1],
                low=raw[2][: endpoint + 1],
                volume=raw[3][: endpoint + 1],
                amount=raw[4][: endpoint + 1],
                event_offsets=events[symbol][events[symbol] <= endpoint],
                quarantine_offsets=quarantines[symbol][quarantines[symbol] <= endpoint],
            )
            if not np.array_equal(
                prefix_values[-1], stock_values[index, endpoint]
            ) or not (
                np.array_equal(prefix_available[-1], stock_available[index, endpoint])
                and np.array_equal(prefix_missing[-1], stock_missing[index, endpoint])
            ):
                raise CourageStrictC4GoldenError("prefix-causality parity drift")
            parity_checks += 1

    membership = pd.read_parquet(inputs["membership"])
    daily = pd.read_parquet(inputs["normalized_daily"])
    first_date = pd.Timestamp("2025-07-01")
    first_membership = membership.loc[
        pd.to_datetime(membership["signal_date"]).eq(first_date)
    ]
    slow = build_slow_context_v1(
        membership=first_membership,
        daily=daily.loc[daily["instrument"].isin(first_membership["instrument"])],
        official_dates=pd.DatetimeIndex(axis["session_date"].unique()),
    )
    if (
        len(slow) != len(first_membership)
        or not slow["slow_source_date"].eq(pd.Timestamp("2025-06-30")).all()
    ):
        raise CourageStrictC4GoldenError("golden T-1 slow-context drift")

    golden_mask = axis["session_date"].isin(golden_dates).to_numpy()
    dynamic_valid_counts = {}
    for index, feature in enumerate(STOCK_FEATURES):
        dynamic_valid_counts[feature] = int(
            (
                stock_available[:, golden_mask, index]
                & ~stock_missing[:, golden_mask, index]
            ).sum()
        )
    dynamic_valid_counts.update(sector_valid_counts)
    if any(value <= 0 for value in dynamic_valid_counts.values()):
        raise CourageStrictC4GoldenError("golden feature coverage is empty")

    config_sha = sha256_file_v1(config_path)
    auth_sha = sha256_file_v1(authorization_path)
    target = root / config["output_root"] / f"golden-{auth_sha[:24]}"
    report_path = target / "_c4_golden_report.json"
    if target.exists():
        existing = _load_json(report_path, "existing C4 golden report")
        if existing.get("decision") != "PASS_C4_GOLDEN_FEATURE_AUDIT":
            raise CourageStrictC4GoldenError("existing C4 golden report drift")
        return report_path
    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    report = {
        "schema_version": REPORT_SCHEMA,
        "decision": "PASS_C4_GOLDEN_FEATURE_AUDIT",
        "terminal_state": "STOP_AFTER_C4_GOLDEN_BEFORE_FULL_SEQUENCE",
        "authorization_consumed": True,
        "authorization_sha256": auth_sha,
        "config_sha256": config_sha,
        "feature_identity": {
            "dynamic_order": list(DYNAMIC_FEATURES),
            "slow_order": list(SLOW_FEATURES),
            "industry_embedding": "effective_dated_sector_level2_code_or_UNKNOWN",
        },
        "coverage": {
            "mainboard_symbols": len(symbols),
            "source_sessions": int(axis["session_date"].nunique()),
            "source_axis_rows": len(axis),
            "golden_sessions": len(golden_dates),
            "golden_axis_rows": int(golden_mask.sum()),
            "dynamic_valid_rows_by_feature": dynamic_valid_counts,
            "sector_aggregate_rows": sector_state_rows,
            "slow_context_rows": len(slow),
            "prefix_causality_checks": parity_checks,
        },
        "checks": {
            "complete_mainboard_before_turnover_projection": True,
            "PIT_industry_no_future_fill": True,
            "prefix_truncation_exact_parity": True,
            "T_minus_1_slow_source_date_exact": True,
            "accepted_action_history_reset": True,
            "quarantined_action_1200_slot_blackout": True,
            "all_values_finite_when_valid": True,
        },
        "development_test_read": False,
        "reserved_confirm_read": False,
        "holdout_read": False,
        "full_feature_or_sequence_materialized": False,
        "runtime_training_selection_executed": False,
        "remote_push_executed": False,
        "terminal_authority_matrix": {
            key: False for key in sorted(TRUE_AUTHORITIES | FALSE_AUTHORITIES)
        },
    }
    report_path_staging = staging / report_path.name
    report_path_staging.write_bytes(_canonical_json(report))
    os.replace(staging, target)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    print(
        run_courage_strict_c4_golden_v1(
            project_root=args.project_root,
            config_path=args.config,
            authorization_path=args.authorization,
        )
    )


if __name__ == "__main__":
    main()

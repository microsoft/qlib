"""C3-A strict T-1 turnover membership and legal signal grid."""

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


class CourageStrictC3MembershipError(RuntimeError):
    """Raised when C3 membership or signal identity drifts."""


CONFIG_SCHEMA: Final[str] = "courage_strict_c3_membership_config_v1"
AUTH_SCHEMA: Final[str] = "courage_strict_c3_membership_authorization_v1"
MANIFEST_SCHEMA: Final[str] = "courage_strict_c3_membership_manifest_v1"
OPERATOR_STATEMENT: Final[str] = (
    "用户授权在当前Qlib仓库完成courage_strict_v1全部代码、数据构造、训练和评测。"
)
TRUE_AUTHORITIES: Final[set[str]] = {
    "execution_authorized",
    "accepted_C2_read_authorized",
    "turnover_membership_materialization_authorized",
    "signal_grid_materialization_authorized",
    "C3_evidence_write_authorized",
}
FALSE_AUTHORITIES: Final[set[str]] = {
    "minute_bar_record_read_authorized",
    "label_build_authorized",
    "feature_build_authorized",
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
        raise CourageStrictC3MembershipError(f"unsafe or missing {label}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CourageStrictC3MembershipError(f"{label} must be object")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = (
        (root / value).resolve()
        if not Path(value).is_absolute()
        else Path(value).resolve()
    )
    if path.is_symlink() or not path.is_file():
        raise CourageStrictC3MembershipError(f"unsafe or missing input: {value}")
    return path


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()


def build_membership_v1(
    *,
    scope: pd.DataFrame,
    daily: pd.DataFrame,
    status: pd.DataFrame,
    full_open_dates: np.ndarray,
    signal_dates: pd.DatetimeIndex,
    role_by_date: dict[pd.Timestamp, str],
) -> pd.DataFrame:
    symbols = sorted(scope["instrument"].astype(str))
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(daily["session_date"]).unique()))
    turnover = daily.pivot(
        index="session_date", columns="instrument", values="turnover_rate_f_percent"
    ).reindex(index=dates, columns=symbols)
    rolling_mean = turnover.rolling(60, min_periods=50).mean().shift(1)
    rolling_count = turnover.rolling(60, min_periods=1).count().shift(1)
    excluded = (
        status.pivot(
            index="session_date",
            columns="instrument",
            values="membership_status_fail_closed_excluded",
        )
        .reindex(index=dates, columns=symbols)
        .shift(1)
        .astype("boolean")
        .fillna(True)
        .astype(bool)
    )
    scope_indexed = scope.set_index("instrument").reindex(symbols)
    list_date = pd.to_datetime(scope_indexed["list_date"]).to_numpy("datetime64[D]")
    delist_date = pd.to_datetime(scope_indexed["delist_date"]).to_numpy("datetime64[D]")
    first_open = np.searchsorted(full_open_dates, list_date, side="left")
    parts: list[pd.DataFrame] = []
    for signal_date in signal_dates:
        day = np.datetime64(signal_date.date())
        open_position = np.searchsorted(full_open_dates, day, side="left")
        mean = rolling_mean.loc[signal_date].to_numpy(dtype=float)
        count = rolling_count.loc[signal_date].to_numpy(dtype=float)
        eligible = (
            np.isfinite(mean)
            & (mean >= 5.0)
            & (mean <= 15.0)
            & (count >= 50)
            & ~excluded.loc[signal_date].to_numpy(dtype=bool)
            & ((open_position - first_open) >= 120)
            & (list_date <= day)
            & (np.isnat(delist_date) | (delist_date >= day))
        )
        indexes = np.flatnonzero(eligible)
        prior_dates = dates[dates < signal_date][-60:]
        parts.append(
            pd.DataFrame(
                {
                    "instrument": np.asarray(symbols, dtype=object)[indexes],
                    "signal_date": signal_date,
                    "role": role_by_date[signal_date],
                    "turnover_mean_60_percent": mean[indexes],
                    "turnover_valid_observations": count[indexes].astype(np.int16),
                    "turnover_window_start": prior_dates[0],
                    "turnover_window_end": prior_dates[-1],
                    "status_source_date": dates[dates < signal_date][-1],
                    "listing_official_sessions_before_signal": (
                        open_position - first_open[indexes]
                    ).astype(np.int32),
                    "membership_lower_inclusive": 5.0,
                    "membership_upper_inclusive": 15.0,
                    "strict_T_minus_1": True,
                }
            )
        )
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["instrument", "signal_date"]).any():
        raise CourageStrictC3MembershipError("duplicate membership key")
    return result.sort_values(
        ["signal_date", "instrument"], kind="mergesort"
    ).reset_index(drop=True)


def build_signal_grid_v1(
    axis: pd.DataFrame, signal_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    frame = axis.loc[axis["exchange"].eq("SSE")].copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    frame = frame.sort_values(
        ["session_date", "minute_slot"], kind="mergesort"
    ).reset_index(drop=True)
    frame["global_minute_offset"] = np.arange(len(frame), dtype=np.int64)
    starts_ns = (
        pd.to_datetime(frame["minute_window_start"], utc=True)
        .astype("int64")
        .to_numpy()
    )
    parts: list[pd.DataFrame] = []
    for date in signal_dates:
        day = frame.loc[frame["session_date"].eq(date)].copy()
        local_ready = pd.to_datetime(day["feature_ready_time"], utc=True).dt.tz_convert(
            "Asia/Shanghai"
        )
        clock_minutes = local_ready.dt.hour * 60 + local_ready.dt.minute
        legal = day.loc[
            clock_minutes.between(9 * 60 + 30, 11 * 60 + 30)
            | clock_minutes.between(13 * 60, 15 * 60)
        ].copy()
        legal["legal_state_index"] = np.arange(len(legal), dtype=np.int16)
        selected = legal.loc[legal["legal_state_index"].mod(5).eq(0)].copy()
        signal_ns = pd.to_datetime(selected["feature_ready_time"], utc=True).astype(
            "int64"
        )
        selected["entry_global_minute_offset"] = np.searchsorted(
            starts_ns, signal_ns.to_numpy(), side="left"
        ).astype(np.int64)
        selected["signal_time"] = selected["feature_ready_time"]
        parts.append(
            selected.loc[
                :,
                [
                    "session_date",
                    "signal_time",
                    "minute_slot",
                    "global_minute_offset",
                    "legal_state_index",
                    "entry_global_minute_offset",
                ],
            ]
        )
    result = pd.concat(parts, ignore_index=True)
    if result.groupby("session_date").size().ne(48).any():
        raise CourageStrictC3MembershipError("selected signal count is not 48/day")
    if result.duplicated(["signal_time"]).any():
        raise CourageStrictC3MembershipError("duplicate signal time")
    return result


def _validate_controls(
    root: Path, config_path: Path, auth_path: Path
) -> dict[str, Any]:
    config, auth = (
        _load_json(config_path, "C3 config"),
        _load_json(auth_path, "C3 auth"),
    )
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("status") != "FROZEN_NOT_AUTHORIZED"
    ):
        raise CourageStrictC3MembershipError("config drift")
    if any(value is not False for value in config["authority_matrix"].values()):
        raise CourageStrictC3MembershipError("config authority drift")
    for label, identity in config["inputs"].items():
        if sha256_file_v1(_resolve(root, identity["path"])) != identity["sha256"]:
            raise CourageStrictC3MembershipError(f"input drift: {label}")
    if (
        auth.get("schema_version") != AUTH_SCHEMA
        or auth.get("operator_statement") != OPERATOR_STATEMENT
    ):
        raise CourageStrictC3MembershipError("auth identity drift")
    if (
        auth.get("operator_statement_sha256")
        != hashlib.sha256(OPERATOR_STATEMENT.encode()).hexdigest()
    ):
        raise CourageStrictC3MembershipError("operator SHA drift")
    if (
        auth.get("one_time_authorization") is not True
        or auth.get("consumed_when_manifest_written") is not True
    ):
        raise CourageStrictC3MembershipError("auth lifecycle drift")
    if auth.get("config") != {
        "path": config_path.relative_to(root).as_posix(),
        "sha256": sha256_file_v1(config_path),
    }:
        raise CourageStrictC3MembershipError("auth config drift")
    runner = Path(__file__).resolve()
    if auth.get("runner") != {
        "path": runner.relative_to(root).as_posix(),
        "sha256": sha256_file_v1(runner),
    }:
        raise CourageStrictC3MembershipError("auth runner drift")
    authorities = auth.get("authorities")
    if (
        not isinstance(authorities, dict)
        or set(authorities) != TRUE_AUTHORITIES | FALSE_AUTHORITIES
    ):
        raise CourageStrictC3MembershipError("auth key drift")
    if any(authorities[key] is not True for key in TRUE_AUTHORITIES) or any(
        authorities[key] is not False for key in FALSE_AUTHORITIES
    ):
        raise CourageStrictC3MembershipError("auth scope drift")
    return config


def run_courage_strict_c3_membership_v1(
    *, project_root: Path, config_path: Path, authorization_path: Path
) -> Path:
    root = project_root.resolve()
    config_file, auth_file = config_path.resolve(), authorization_path.resolve()
    config = _validate_controls(root, config_file, auth_file)
    config_sha, auth_sha = sha256_file_v1(config_file), sha256_file_v1(auth_file)
    target = root / config["output_root"] / f"authorization-{auth_sha[:24]}"
    manifest_path = target / "_c3_membership_manifest.json"
    if manifest_path.is_file():
        manifest = _load_json(manifest_path, "existing C3 membership manifest")
        if (
            manifest.get("decision") != "PASS_C3_A_MEMBERSHIP_AND_SIGNAL_GRID"
            or manifest.get("config_sha256") != config_sha
        ):
            raise CourageStrictC3MembershipError("existing manifest drift")
        for item in manifest["files"]:
            if sha256_file_v1(target / item["path"]) != item["sha256"]:
                raise CourageStrictC3MembershipError("existing output drift")
        return manifest_path

    paths = {
        name: _resolve(root, identity["path"])
        for name, identity in config["inputs"].items()
    }
    scope = pq.read_table(paths["mainboard_scope"]).to_pandas(ignore_metadata=True)
    daily = pq.read_table(
        paths["normalized_daily"],
        columns=["instrument", "session_date", "turnover_rate_f_percent"],
    ).to_pandas(ignore_metadata=True)
    status = pq.read_table(
        paths["normalized_status"],
        columns=[
            "instrument",
            "session_date",
            "membership_status_fail_closed_excluded",
        ],
    ).to_pandas(ignore_metadata=True)
    symbols = set(scope["instrument"])
    daily = daily.loc[daily["instrument"].isin(symbols)].copy()
    status = status.loc[status["instrument"].isin(symbols)].copy()
    daily["session_date"] = pd.to_datetime(daily["session_date"]).dt.normalize()
    status["session_date"] = pd.to_datetime(status["session_date"]).dt.normalize()
    axis = pq.read_table(paths["official_axis"]).to_pandas(ignore_metadata=True)
    axis_dates = pd.DatetimeIndex(sorted(pd.to_datetime(axis["session_date"]).unique()))
    train = axis_dates[
        (axis_dates >= pd.Timestamp("2025-07-01"))
        & (axis_dates < pd.Timestamp("2026-03-02"))
    ][:-1]
    valid = axis_dates[
        (axis_dates >= pd.Timestamp("2026-03-02"))
        & (axis_dates < pd.Timestamp("2026-04-01"))
    ][:-1]
    signal_dates = train.append(valid)
    role_by_date = {date: "train" for date in train} | {date: "valid" for date in valid}
    calendar = pq.read_table(
        paths["full_calendar"], columns=["trade_date", "is_open"]
    ).to_pandas(ignore_metadata=True)
    open_dates = np.sort(
        pd.to_datetime(calendar.loc[calendar["is_open"].eq(1), "trade_date"]).to_numpy(
            dtype="datetime64[D]"
        )
    )
    membership = build_membership_v1(
        scope=scope,
        daily=daily,
        status=status,
        full_open_dates=open_dates,
        signal_dates=signal_dates,
        role_by_date=role_by_date,
    )
    grid = build_signal_grid_v1(axis, signal_dates)
    role_counts = membership.groupby("role").size().to_dict()
    observed = {
        "signal_dates": len(signal_dates),
        "train_dates": len(train),
        "valid_dates": len(valid),
        "membership_rows": len(membership),
        "train_membership_rows": int(role_counts["train"]),
        "valid_membership_rows": int(role_counts["valid"]),
        "selected_signal_states": len(grid),
        "samples": int(len(membership) * 48),
        "train_samples": int(role_counts["train"] * 48),
        "valid_samples": int(role_counts["valid"] * 48),
    }
    if observed != config["expected_counts"]:
        raise CourageStrictC3MembershipError(f"C3-A count drift: {observed}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.{uuid.uuid4().hex}.staging"
    staging.mkdir()
    try:
        frames = {"membership.parquet": membership, "signal_grid.parquet": grid}
        files = []
        for name, frame in frames.items():
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
            "decision": "PASS_C3_A_MEMBERSHIP_AND_SIGNAL_GRID",
            "terminal_state": "STOP_AFTER_C3_A_BEFORE_LABELS",
            "config_sha256": config_sha,
            "authorization_sha256": auth_sha,
            "authorization_consumed": True,
            "membership_contract": config["membership_contract"],
            "signal_contract": config["signal_contract"],
            "coverage": observed,
            "files": files,
            "minute_bar_records_read": False,
            "development_test_read": False,
            "training_executed": False,
            "remote_push_executed": False,
        }
        (staging / manifest_path.name).write_bytes(_canonical_json(manifest))
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
    print(
        run_courage_strict_c3_membership_v1(
            project_root=args.project_root,
            config_path=args.config,
            authorization_path=args.authorization,
        )
    )


if __name__ == "__main__":
    main()

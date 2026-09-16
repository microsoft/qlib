"""C2-R2 normalization for effective-dated security status.

This module repairs only the C2-G06 input.  It deliberately does not build a
universe or a training dataset.  The source ``is_st`` flag is retained, while
formal delisting-risk status is represented as a nullable, independently
auditable field.  Unknown risk subtype is fail-closed for membership.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html
import io
import json
import os
import re
import shutil
import uuid
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import requests

from qlib.contrib.data.courage_strict_v1.minute_courage_strict_c2_identity_v1 import (
    sha256_file_v1,
)


class CourageStrictC2StatusR2Error(RuntimeError):
    """Raised when the bounded G06 remediation cannot be proven."""


CONFIG_SCHEMA: Final[str] = "courage_strict_c2_status_r2_config_v1"
AUTH_SCHEMA: Final[str] = "courage_strict_c2_status_r2_authorization_v1"
MANIFEST_SCHEMA: Final[str] = "courage_strict_c2_status_r2_manifest_v1"
OPERATOR_STATEMENT: Final[str] = (
    "请把下面的目标提交到著目录空间下的持久Codex.sh。确认任务已经启动并给出任务编号后再回复我。"
    "我提供给你所有权限，你需要充分利用资源，并且以较快的速度完成C0-C8任务。"
)
TRUE_AUTHORITIES: Final[set[str]] = {
    "execution_authorized",
    "official_rule_snapshot_authorized",
    "official_announcement_metadata_snapshot_authorized",
    "source_record_read_authorized",
    "c2_status_normalization_authorized",
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
STATUS_COLUMNS: Final[tuple[str, ...]] = (
    "instrument",
    "session_date",
    "status_available_at",
    "is_ST",
    "has_delisting_risk",
    "delisting_risk_known",
    "is_suspended",
    "limit_up_price",
    "limit_down_price",
    "terminal_delisting_event_available_at",
    "membership_status_fail_closed_excluded",
    "same_session_intraday_use_authorized",
    "source_is_st",
    "source_suspend_type",
)


@dataclass(frozen=True)
class TerminationEventV1:
    instrument: str
    announcement_time: pd.Timestamp
    announcement_id: str
    title: str


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CourageStrictC2StatusR2Error(f"unsafe or missing {label}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CourageStrictC2StatusR2Error(f"invalid {label}") from exc
    if not isinstance(value, dict):
        raise CourageStrictC2StatusR2Error(f"{label} must be an object")
    return value


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


def _resolve(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CourageStrictC2StatusR2Error("configured path escapes project") from exc
    return path


def _validate_controls(
    *, root: Path, config_path: Path, authorization_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path, "C2-R2 status config")
    authorization = _load_json(authorization_path, "C2-R2 status authorization")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise CourageStrictC2StatusR2Error("config schema drift")
    if config.get("status") != "FROZEN_NOT_AUTHORIZED":
        raise CourageStrictC2StatusR2Error("config status drift")
    if config.get("record_read_interval") != ["2025-04-01", "2026-04-01"]:
        raise CourageStrictC2StatusR2Error("record interval drift")
    if any(value is not False for value in config.get("authority_matrix", {}).values()):
        raise CourageStrictC2StatusR2Error("config authorities must remain false")
    for label, identity in config["control_identities"].items():
        if sha256_file_v1(_resolve(root, identity["path"])) != identity["sha256"]:
            raise CourageStrictC2StatusR2Error(f"control identity drift: {label}")
    for label, identity in config["source_inputs"].items():
        path = Path(identity["absolute_path"])
        if path.is_symlink() or not path.is_file():
            raise CourageStrictC2StatusR2Error(f"unsafe source: {label}")
        if sha256_file_v1(path) != identity["sha256"]:
            raise CourageStrictC2StatusR2Error(f"source identity drift: {label}")

    if authorization.get("schema_version") != AUTH_SCHEMA:
        raise CourageStrictC2StatusR2Error("authorization schema drift")
    if authorization.get("one_time_authorization") is not True:
        raise CourageStrictC2StatusR2Error("authorization must be one-time")
    if authorization.get("consumed_when_manifest_written") is not True:
        raise CourageStrictC2StatusR2Error("authorization consumption drift")
    if authorization.get("operator_statement") != OPERATOR_STATEMENT:
        raise CourageStrictC2StatusR2Error("operator statement drift")
    statement_sha = hashlib.sha256(OPERATOR_STATEMENT.encode("utf-8")).hexdigest()
    if authorization.get("operator_statement_sha256") != statement_sha:
        raise CourageStrictC2StatusR2Error("operator statement SHA drift")
    if authorization.get("config") != {
        "path": config_path.relative_to(root).as_posix(),
        "sha256": sha256_file_v1(config_path),
    }:
        raise CourageStrictC2StatusR2Error("authorization config identity drift")
    runner = Path(__file__).resolve()
    if authorization.get("runner") != {
        "path": runner.relative_to(root).as_posix(),
        "sha256": sha256_file_v1(runner),
    }:
        raise CourageStrictC2StatusR2Error("authorization runner identity drift")
    authorities = authorization.get("authorities")
    if not isinstance(authorities, dict) or set(authorities) != (
        TRUE_AUTHORITIES | FALSE_AUTHORITIES
    ):
        raise CourageStrictC2StatusR2Error("authority key drift")
    if any(authorities[key] is not True for key in TRUE_AUTHORITIES):
        raise CourageStrictC2StatusR2Error("required C2-R2 authority is false")
    if any(authorities[key] is not False for key in FALSE_AUTHORITIES):
        raise CourageStrictC2StatusR2Error("forbidden authority is true")
    return config, authorization


def _clean_title(value: str | None) -> str:
    return re.sub(r"\s+", "", html.unescape(re.sub(r"<[^>]+>", "", value or "")))


def classify_termination_events_v1(
    announcements: Iterable[dict[str, Any]], *, admitted_instruments: set[str]
) -> tuple[TerminationEventV1, ...]:
    """Select only affirmative stock termination decisions or delisting notices."""

    accepted_tokens = (
        "收到股票终止上市决定",
        "股票终止上市暨摘牌",
        "公司股票终止上市的公告",
    )
    forbidden_tokens = (
        "可能",
        "风险提示",
        "申请",
        "意见",
        "听证",
        "事先告知",
        "进展",
        "撤回",
        "暂缓",
        "债券",
    )
    by_key: dict[tuple[str, int, str], TerminationEventV1] = {}
    for item in announcements:
        code = str(item.get("secCode") or "")
        suffix = ".SH" if code.startswith("6") else ".SZ"
        instrument = f"{code}{suffix}"
        title = _clean_title(item.get("announcementTitle"))
        if instrument not in admitted_instruments:
            continue
        if not any(token in title for token in accepted_tokens):
            continue
        if any(token in title for token in forbidden_tokens):
            continue
        timestamp_ms = item.get("announcementTime")
        if not isinstance(timestamp_ms, int):
            continue
        timestamp = pd.Timestamp(timestamp_ms, unit="ms", tz="Asia/Shanghai")
        event = TerminationEventV1(
            instrument=instrument,
            announcement_time=timestamp,
            announcement_id=str(item.get("announcementId") or ""),
            title=title,
        )
        by_key[(instrument, timestamp_ms, event.announcement_id)] = event
    return tuple(
        sorted(by_key.values(), key=lambda x: (x.instrument, x.announcement_time))
    )


def build_status_table_v1(
    *,
    daily: pd.DataFrame,
    session_ready: pd.Series,
    termination_events: Iterable[TerminationEventV1],
) -> pd.DataFrame:
    """Build the canonical nullable-risk table from authorized source rows."""

    required = {
        "trade_date",
        "ts_code",
        "is_st",
        "suspend_type",
        "up_limit",
        "down_limit",
    }
    if required - set(daily.columns):
        raise CourageStrictC2StatusR2Error("daily status columns missing")
    frame = daily.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    if frame.duplicated(["ts_code", "trade_date"]).any():
        raise CourageStrictC2StatusR2Error("duplicate status key")
    if not frame["is_st"].isin([0, 1]).all():
        raise CourageStrictC2StatusR2Error("invalid is_st domain")
    ready_by_date = {
        pd.Timestamp(key).normalize(): pd.Timestamp(value)
        for key, value in session_ready.items()
    }
    frame["status_available_at"] = frame["trade_date"].map(ready_by_date)
    if frame["status_available_at"].isna().any():
        raise CourageStrictC2StatusR2Error("status date absent from official axis")

    event_by_instrument: dict[str, list[TerminationEventV1]] = {}
    for event in termination_events:
        event_by_instrument.setdefault(event.instrument, []).append(event)
    terminal_at: list[pd.Timestamp | pd.NaT] = []
    for instrument, date in zip(frame["ts_code"], frame["trade_date"], strict=True):
        eligible = [
            event.announcement_time
            for event in event_by_instrument.get(str(instrument), [])
            if event.announcement_time <= ready_by_date[pd.Timestamp(date).normalize()]
        ]
        terminal_at.append(max(eligible) if eligible else pd.NaT)
    frame["terminal_delisting_event_available_at"] = terminal_at
    is_st = frame["is_st"].astype(bool)
    terminal = frame["terminal_delisting_event_available_at"].notna()
    risk = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    risk.loc[~is_st] = False
    risk.loc[terminal] = True
    frame["has_delisting_risk"] = risk
    frame["delisting_risk_known"] = risk.notna()
    frame["membership_status_fail_closed_excluded"] = is_st | risk.fillna(True).astype(
        bool
    )
    frame["same_session_intraday_use_authorized"] = False
    frame["instrument"] = frame["ts_code"].astype(str)
    frame["session_date"] = frame["trade_date"].dt.date
    frame["is_ST"] = is_st
    frame["is_suspended"] = ~frame["suspend_type"].eq("N")
    frame["limit_up_price"] = pd.to_numeric(frame["up_limit"], errors="coerce")
    frame["limit_down_price"] = pd.to_numeric(frame["down_limit"], errors="coerce")
    frame.loc[~(frame["limit_up_price"] > 0), "limit_up_price"] = pd.NA
    frame.loc[~(frame["limit_down_price"] > 0), "limit_down_price"] = pd.NA
    frame["source_is_st"] = frame["is_st"].astype("int8")
    frame["source_suspend_type"] = frame["suspend_type"].astype(str)
    result = frame.loc[:, list(STATUS_COLUMNS)].sort_values(
        ["session_date", "instrument"], kind="stable"
    )
    if result["same_session_intraday_use_authorized"].any():
        raise CourageStrictC2StatusR2Error("same-session use cannot be authorized")
    if result.loc[~result["delisting_risk_known"], "is_ST"].ne(True).any():
        raise CourageStrictC2StatusR2Error("risk unknown outside ST rows")
    if (
        result.loc[
            ~result["delisting_risk_known"], "membership_status_fail_closed_excluded"
        ]
        .ne(True)
        .any()
    ):
        raise CourageStrictC2StatusR2Error("unknown risk was not excluded")
    return result.reset_index(drop=True)


def _download(url: str) -> bytes:
    last: Exception | None = None
    for _ in range(3):
        try:
            response = requests.get(
                url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60
            )
            response.raise_for_status()
            if len(response.content) < 1_000:
                raise CourageStrictC2StatusR2Error("official source unexpectedly small")
            return response.content
        except (requests.RequestException, CourageStrictC2StatusR2Error) as exc:
            last = exc
    raise CourageStrictC2StatusR2Error(
        f"official source download failed: {url}"
    ) from last


def _extract_rule_text(content: bytes, suffix: str) -> str:
    if suffix == ".docx":
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            raw = archive.read("word/document.xml").decode("utf-8")
        return re.sub(r"<[^>]+>", "", raw)
    if suffix in {".html", ".shtml"}:
        return re.sub(r"<[^>]+>", "", content.decode("utf-8", errors="ignore"))
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n".join(
            page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages
        )
    raise CourageStrictC2StatusR2Error("unsupported official rule format")


def _snapshot_rules(stage: Path, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in sources:
        content = _download(source["url"])
        relative = Path("official_rules") / source["filename"]
        target = stage / relative
        _write_bytes_exclusive(target, content)
        text = re.sub(r"\s+", "", _extract_rule_text(content, target.suffix.lower()))
        if "退市风险警示" not in text or "*ST" not in text:
            raise CourageStrictC2StatusR2Error(
                "official rule does not prove *ST semantics"
            )
        records.append(
            {
                "exchange": source["exchange"],
                "edition": source["edition"],
                "url": source["url"],
                "path": relative.as_posix(),
                "sha256": sha256_file_v1(target),
                "bytes": target.stat().st_size,
                "semantic_assertion": "formal_delisting_risk_warning_implies_star_ST_name_prefix",
            }
        )
    return records


def _month_spans(start: str, end: str) -> list[str]:
    first = pd.Timestamp(start)
    last = pd.Timestamp(end) - pd.Timedelta(days=1)
    periods = pd.period_range(first, last, freq="M")
    return [
        f"{period.start_time.date().isoformat()}~{period.end_time.date().isoformat()}"
        for period in periods
    ]


def _fetch_cninfo_month(span: str) -> list[dict[str, Any]]:
    url = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"}
    base = {
        "pageNum": "1",
        "pageSize": "30",
        "column": "szse",
        "tabName": "fulltext",
        "plate": "",
        "stock": "",
        "searchkey": "终止上市",
        "secid": "",
        "category": "",
        "trade": "",
        "seDate": span,
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }

    def page(number: int) -> dict[str, Any]:
        response = requests.post(
            url, data={**base, "pageNum": str(number)}, headers=headers, timeout=60
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise CourageStrictC2StatusR2Error("invalid CNInfo response")
        return payload

    first = page(1)
    values = list(first.get("announcements") or [])
    pages = int(first.get("totalpages") or 1)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(8, max(1, pages - 1))
    ) as pool:
        for payload in pool.map(page, range(2, pages + 1)):
            values.extend(payload.get("announcements") or [])
    return values


def _snapshot_announcements(
    stage: Path, *, start: str, end: str
) -> list[dict[str, Any]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        groups = list(pool.map(_fetch_cninfo_month, _month_spans(start, end)))
    by_id = {
        str(item["announcementId"]): item
        for group in groups
        for item in group
        if item.get("announcementId")
    }
    values = sorted(
        by_id.values(),
        key=lambda item: (
            int(item.get("announcementTime") or 0),
            str(item.get("secCode") or ""),
            str(item.get("announcementId") or ""),
        ),
    )
    _write_bytes_exclusive(
        stage / "cninfo_termination_query.json", _canonical_json_bytes(values)
    )
    return values


def _session_ready(axis_path: Path) -> pd.Series:
    axis = pq.read_table(
        axis_path, columns=["session_date", "exchange", "feature_ready_time"]
    ).to_pandas()
    grouped = axis.groupby(["session_date", "exchange"])["feature_ready_time"].max()
    by_date = grouped.groupby(level=0).agg(lambda values: values.iloc[0])
    exchange_match = grouped.groupby(level=0).nunique().eq(1)
    if not exchange_match.all():
        raise CourageStrictC2StatusR2Error("exchange final-ready time mismatch")
    return by_date


def _read_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, set[str], pd.Series]:
    start, end = config["record_read_interval"]
    master_path = Path(config["source_inputs"]["security_master"]["absolute_path"])
    master = pq.read_table(
        master_path,
        columns=[
            "ts_code",
            "exchange",
            "market",
            "curr_type",
            "list_date",
            "delist_date",
        ],
    ).to_pandas()
    admitted = master.loc[
        master["exchange"].isin(["SSE", "SZSE"])
        & master["curr_type"].eq("CNY")
        & master["market"].isin(["主板", "创业板", "科创板"])
        & master["list_date"].lt(pd.Timestamp(end))
        & (
            master["delist_date"].isna() | master["delist_date"].ge(pd.Timestamp(start))
        ),
        "ts_code",
    ].astype(str)
    instruments = set(admitted)
    dataset = ds.dataset(
        config["source_inputs"]["stock_daily"]["absolute_path"], format="parquet"
    )
    table = dataset.to_table(
        columns=[
            "trade_date",
            "ts_code",
            "is_st",
            "suspend_type",
            "up_limit",
            "down_limit",
        ],
        filter=(ds.field("trade_date") >= pd.Timestamp(start).to_pydatetime())
        & (ds.field("trade_date") < pd.Timestamp(end).to_pydatetime())
        & ds.field("ts_code").isin(sorted(instruments)),
    )
    daily = table.to_pandas()
    if {"trade_date", "ts_code"}.issubset(daily.index.names):
        daily = daily.reset_index()
    axis_path = Path(config["source_inputs"]["official_axis"]["absolute_path"])
    return daily, instruments, _session_ready(axis_path)


def _manifest_reuse(path: Path, *, config_sha: str, auth_sha: str) -> Path | None:
    if not path.is_file():
        return None
    manifest = _load_json(path, "existing C2-R2 status manifest")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("decision") != "PASS_C2_R2_EFFECTIVE_DATED_STATUS_NORMALIZATION"
        or manifest.get("config_sha256") != config_sha
        or manifest.get("authorization_sha256") != auth_sha
        or manifest.get("authorization_consumed") is not True
        or manifest.get("all_downstream_authorities_false") is not True
    ):
        raise CourageStrictC2StatusR2Error("existing manifest drift")
    root = path.parent
    records = manifest.get("files")
    if not isinstance(records, list):
        raise CourageStrictC2StatusR2Error("existing file records missing")
    expected = {path.name}
    for record in records:
        item = root / record["path"]
        if item.is_symlink() or not item.is_file():
            raise CourageStrictC2StatusR2Error("existing artifact missing")
        if sha256_file_v1(item) != record["sha256"]:
            raise CourageStrictC2StatusR2Error("existing artifact SHA drift")
        expected.add(record["path"])
    actual = {
        item.relative_to(root).as_posix() for item in root.rglob("*") if item.is_file()
    }
    if actual != expected:
        raise CourageStrictC2StatusR2Error("existing artifact inventory drift")
    return path


def run_courage_strict_c2_status_r2_v1(
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
    final = _resolve(root, config["output_root"]) / f"authorization-{auth_sha[:24]}"
    manifest_path = final / "_status_r2_manifest.json"
    existing = _manifest_reuse(manifest_path, config_sha=config_sha, auth_sha=auth_sha)
    if existing is not None:
        return existing
    if final.exists():
        raise CourageStrictC2StatusR2Error("incomplete final target exists")
    stage = final.with_name(f".{final.name}.{uuid.uuid4().hex}.staging")
    stage.mkdir(parents=True)
    try:
        rule_records = _snapshot_rules(stage, config["official_rule_sources"])
        announcements = _snapshot_announcements(
            stage,
            start=config["announcement_query_interval"][0],
            end=config["announcement_query_interval"][1],
        )
        daily, instruments, ready = _read_inputs(config)
        events = classify_termination_events_v1(
            announcements, admitted_instruments=instruments
        )
        status = build_status_table_v1(
            daily=daily, session_ready=ready, termination_events=events
        )
        status_path = stage / "effective_dated_status.parquet"
        table = pa.Table.from_pandas(status, preserve_index=False)
        pq.write_table(table, status_path, compression="zstd", row_group_size=100_000)
        announcement_path = stage / "cninfo_termination_query.json"
        files = [
            {
                "kind": "normalized_status",
                "path": status_path.name,
                "sha256": sha256_file_v1(status_path),
                "bytes": status_path.stat().st_size,
            },
            {
                "kind": "official_announcement_metadata_snapshot",
                "path": announcement_path.name,
                "sha256": sha256_file_v1(announcement_path),
                "bytes": announcement_path.stat().st_size,
            },
            *[
                {
                    "kind": "official_exchange_rule_snapshot",
                    "path": record["path"],
                    "sha256": record["sha256"],
                    "bytes": record["bytes"],
                }
                for record in rule_records
            ],
        ]
        risk = status["has_delisting_risk"]
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "decision": "PASS_C2_R2_EFFECTIVE_DATED_STATUS_NORMALIZATION",
            "terminal_state": "STOP_AFTER_C2_R2_STATUS_BEFORE_G06_REAUDIT",
            "config_sha256": config_sha,
            "authorization_sha256": auth_sha,
            "authorization_consumed": True,
            "operator_statement_sha256": authorization["operator_statement_sha256"],
            "record_read_interval": config["record_read_interval"],
            "rows": len(status),
            "session_dates": int(status["session_date"].nunique()),
            "instruments": int(status["instrument"].nunique()),
            "is_ST_rows": int(status["is_ST"].sum()),
            "delisting_risk_true_rows": int(risk.fillna(False).sum()),
            "delisting_risk_unknown_rows": int(risk.isna().sum()),
            "unknown_risk_fail_closed_excluded_rows": int(
                (risk.isna() & status["membership_status_fail_closed_excluded"]).sum()
            ),
            "termination_events": [
                {
                    "instrument": event.instrument,
                    "announcement_time": event.announcement_time.isoformat(),
                    "announcement_id": event.announcement_id,
                    "title": event.title,
                }
                for event in events
            ],
            "rule_evidence": rule_records,
            "semantics": {
                "status_available_at": (
                    "official_session_final_minute_feature_ready_time; "
                    "T_minus_1_membership_use_only"
                ),
                "formal_delisting_risk_false": (
                    "per_official_exchange_rule_only_non_ST_rows_without_terminal_event"
                ),
                "formal_delisting_risk_unknown": (
                    "ST_rows_without_independent_subtype; fail_closed_excluded"
                ),
                "terminal_delisting_event": (
                    "affirmative_CNInfo_stock_termination_decision_or_delisting_notice_"
                    "available_by_session_close"
                ),
                "same_session_intraday_use": False,
                "current_day_tradability": (
                    "must_be_derived_later_from_signal_time_ready_sources; "
                    "this_after_close_table_is_not_admitted_for_same_session_use"
                ),
            },
            "source_inputs": config["source_inputs"],
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
            stage / manifest_path.name, _canonical_json_bytes(manifest)
        )
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, final)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    print(
        run_courage_strict_c2_status_r2_v1(
            project_root=args.project_root,
            config_path=args.config,
            authorization_path=args.authorization,
        )
    )


if __name__ == "__main__":
    main()

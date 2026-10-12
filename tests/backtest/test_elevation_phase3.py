"""Filesystem-free tests for executor logging, task limits, and cache guards."""

import logging

import pytest

from qlib.backtest.executor_logging import log_order_execution
from qlib.data.cache_guards import (
    check_cache_file_budget,
    require_trusted_cache_load,
    resolve_cache_file_uri,
)
from qlib.log_structured import JsonFormatter, get_json_logger
from qlib.workflow.task.batch_limits import (
    cap_task_batch,
    check_task_batch_limits,
    estimate_task_bytes,
)


def test_log_order_execution_emits_json_line(caplog) -> None:
    logger = get_json_logger("executor")
    assert isinstance(logger.handlers[0].formatter, JsonFormatter)
    with caplog.at_level(logging.INFO, logger="qlib.executor"):
        log_order_execution(
            stock_id="SH600000",
            direction="buy",
            trade_price=10.0,
            amount=100.0,
            deal_amount=100.0,
            factor=1.0,
            trade_value=1000.0,
            cash=99000.0,
            run_id="phase3",
        )
    assert caplog.records, "expected a structured executor log record"
    assert caplog.records[-1].run_id == "phase3"


def test_task_batch_limits_accept_and_reject() -> None:
    tasks = [{"id": 1}, {"id": 2}]
    report = check_task_batch_limits(tasks, max_tasks=4, max_task_bytes=10**6)
    assert report["num_tasks"] == 2
    assert estimate_task_bytes(tasks[0]) > 0
    with pytest.raises(ValueError):
        check_task_batch_limits(tasks, max_tasks=1)
    with pytest.raises(ValueError):
        check_task_batch_limits([], max_tasks=4)
    with pytest.raises(ValueError):
        check_task_batch_limits([b"x" * 100], max_task_bytes=8)
    with pytest.raises(ValueError):
        check_task_batch_limits(tasks, max_tasks=0)
    assert cap_task_batch(tasks * 3, max_tasks=2) == (tasks * 3)[:2]


def test_cache_guards_resolve_and_budget(tmp_path) -> None:
    base = tmp_path / "cache"
    base.mkdir()
    target = base / "handler.pkl"
    target.write_bytes(b"0123456789")
    resolved = resolve_cache_file_uri("file://handler.pkl", base)
    assert resolved == (base / "handler.pkl").resolve()
    with pytest.raises(ValueError):
        resolve_cache_file_uri("http://evil/handler.pkl", base)
    with pytest.raises(ValueError):
        resolve_cache_file_uri("file://../escape.pkl", base)
    assert check_cache_file_budget(target, max_bytes=1024) == 10
    with pytest.raises(MemoryError):
        check_cache_file_budget(target, max_bytes=4)
    with pytest.raises(FileNotFoundError):
        check_cache_file_budget(base / "missing.pkl")
    with pytest.raises(ValueError):
        require_trusted_cache_load(False, target)
    require_trusted_cache_load(True, target)

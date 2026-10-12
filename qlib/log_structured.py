# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Centralized JSON structured logging for backtest, workflow, and serving paths.

Replaces ad-hoc ``print`` statements with a single formatter so log aggregators
can parse strategy runs (Sharpe, drawdown, latency) without regex scraping.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

_JSON_FORMATTER_NAME = "qlib_json"


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON documents."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003 - logging API
        payload: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("run_id", "strategy", "sharpe", "max_drawdown", "latency_ms", "extra"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_json_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a ``qlib.<name>`` logger configured with the JSON formatter."""
    logger = logging.getLogger(f"qlib.{name}")
    logger.setLevel(level)
    if not any(isinstance(handler.formatter, JsonFormatter) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.propagate = False
    return logger


@contextmanager
def log_latency_ms(logger: logging.Logger, event: str, **fields: Any) -> Iterator[None]:
    """Context manager that logs wall-clock latency of a block in milliseconds."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(event, extra={"extra": {"latency_ms": round(elapsed_ms, 3), **fields}})


def log_backtest_metrics(
    logger: logging.Logger,
    *,
    sharpe: Optional[float] = None,
    max_drawdown: Optional[float] = None,
    latency_ms: Optional[float] = None,
    run_id: Optional[str] = None,
    strategy: Optional[str] = None,
) -> None:
    """Emit one structured metrics line for strategy-performance tracking."""
    record_extra: Dict[str, Any] = {}
    if sharpe is not None:
        record_extra["sharpe"] = sharpe
    if max_drawdown is not None:
        record_extra["max_drawdown"] = max_drawdown
    if latency_ms is not None:
        record_extra["latency_ms"] = latency_ms
    if run_id is not None:
        record_extra["run_id"] = run_id
    if strategy is not None:
        record_extra["strategy"] = strategy
    logger.info("backtest.metrics", extra=record_extra)


__all__ = ["JsonFormatter", "get_json_logger", "log_backtest_metrics", "log_latency_ms"]

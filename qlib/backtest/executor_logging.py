# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Structured per-order execution logging for the backtest simulator path.

Additive helper: :class:`qlib.backtest.executor.SimulatorExecutor` keeps its
``verbose`` CLI behavior, but now emits one JSON line per dealt order through
``qlib.executor`` so log aggregators can track execution latency, slippage
inputs, and cash state without scraping ``print`` output.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from qlib.log_structured import get_json_logger

_executor_logger = get_json_logger("executor")


def log_order_execution(
    *,
    stock_id: str,
    direction: str,
    trade_price: float,
    amount: float,
    deal_amount: float,
    factor: Optional[float],
    trade_value: float,
    cash: float,
    start_time: Any = None,
    latency_ms: Optional[float] = None,
    run_id: Optional[str] = None,
) -> None:
    """Emit one structured JSON line describing a single dealt order."""
    extra: Dict[str, Any] = {
        "strategy": f"executor.{direction}",
        "extra": {
            "stock_id": stock_id,
            "trade_price": trade_price,
            "amount": amount,
            "deal_amount": deal_amount,
            "factor": factor,
            "trade_value": trade_value,
            "cash": cash,
            "start_time": str(start_time) if start_time is not None else None,
        },
    }
    if latency_ms is not None:
        extra["latency_ms"] = latency_ms
    if run_id is not None:
        extra["run_id"] = run_id
    _executor_logger.info("executor.order", extra=extra)


__all__ = ["log_order_execution"]

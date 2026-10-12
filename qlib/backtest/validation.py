# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Pure, filesystem-free validation helpers for the backtest execution path.

This module intentionally depends only on in-memory domain objects
(:class:`qlib.backtest.decision.Order`, :class:`qlib.backtest.position.Position`)
so execution-engine logic can be unit-tested without a local ``~/.qlib`` dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from qlib.backtest.decision import Order, OrderDir
from qlib.backtest.position import Position


def validate_order(order: Order) -> None:
    """Validate structural invariants of a single order.

    Parameters
    ----------
    order : Order
        Order to validate.

    Raises
    ------
    ValueError
        If amount, timestamps, or direction are invalid.
    """
    if order.amount < 0:
        raise ValueError(f"order amount must be non-negative, got {order.amount}")
    if order.direction not in (OrderDir.BUY, OrderDir.SELL):
        raise ValueError(f"unsupported order direction: {order.direction}")
    if order.start_time is None or order.end_time is None:
        raise ValueError("order start_time and end_time must both be set")
    start = pd.Timestamp(order.start_time)
    end = pd.Timestamp(order.end_time)
    if end < start:
        raise ValueError(f"order end_time {end} precedes start_time {start}")


def validate_position_consistency(position: Position) -> None:
    """Check cash/amount invariants of an in-memory position.

    Raises
    ------
    ValueError
        If cash is negative/NaN or any holding amount is negative.
    """
    cash = position.get_cash()
    if not cash == cash or cash < 0:  # NaN check without importing math
        raise ValueError(f"position cash is invalid: {cash!r}")
    for stock_id in position.get_stock_list():
        amount = position.get_stock_amount(stock_id)
        if amount < 0:
            raise ValueError(f"negative holding amount for {stock_id}: {amount}")


def sanitize_cache_path(base_dir: Union[str, Path], requested: Union[str, Path]) -> Path:
    """Resolve ``requested`` strictly inside ``base_dir``.

    Guards CSV/binary cache loaders against path-traversal (``..``) attacks
    when cache file names come from untrusted configs.

    Raises
    ------
    ValueError
        If the resolved path escapes ``base_dir``.
    """
    base = Path(base_dir).expanduser().resolve()
    candidate = (base / str(requested)).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"cache path escapes base directory: {requested!r}") from exc
    return candidate


def check_deal_price_expression(expression: str) -> str:
    """Normalize a deal-price field expression (``close`` -> ``$close``)."""
    if not isinstance(expression, str) or not expression:
        raise ValueError("deal price expression must be a non-empty string")
    expression = expression.strip()
    if not expression.startswith("$"):
        expression = "$" + expression
    if any(token in expression for token in (";", "__", "import", "os.", "sys.")):
        raise ValueError(f"suspicious deal price expression: {expression!r}")
    return expression

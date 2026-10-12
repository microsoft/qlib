# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Dataset-free order-matching test through the injected provider seam.

Builds a real :class:`qlib.backtest.exchange.Exchange` against an
:class:`InMemoryQuoteProvider` — no ``qlib.init()``, no ``~/.qlib`` data —
and deals one buy order end to end (limit/suspension checks, price lookup,
cost accounting, position update).
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from qlib.backtest.account import Account
from qlib.backtest.data_provider import InMemoryQuoteProvider
from qlib.backtest.decision import Order, OrderDir
from qlib.backtest.exchange import Exchange


def _synthetic_quote_df() -> "pd.DataFrame":
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    index = pd.MultiIndex.from_product([["SH600000"], dates], names=["instrument", "datetime"])
    n = len(index)
    return pd.DataFrame(
        {
            "$close": [10.0, 10.5, 11.0][:n],
            "$open": [9.9, 10.4, 10.9][:n],
            "$high": [10.2, 10.7, 11.2][:n],
            "$low": [9.8, 10.3, 10.8][:n],
            "$volume": [1_000_000.0, 1_100_000.0, 1_200_000.0][:n],
            "$factor": [1.0, 1.0, 1.0][:n],
            "$change": [0.0, 0.05, 0.047][:n],
        },
        index=index,
    )


def _exchange() -> Exchange:
    provider = InMemoryQuoteProvider(quote_df=_synthetic_quote_df(), codes=["SH600000"])
    return Exchange(
        freq="day",
        start_time="2020-01-02",
        end_time="2020-01-06",
        codes=["SH600000"],
        deal_price="$close",
        limit_threshold=None,
        open_cost=0.0005,
        close_cost=0.0015,
        min_cost=5.0,
        data_provider=provider,
    )


def test_exchange_deals_buy_order_without_dataset() -> None:
    exchange = _exchange()
    assert exchange.is_stock_tradable(
        "SH600000", pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03"), OrderDir.BUY
    )
    account = Account(init_cash=1_000_000.0, position_dict={}, freq="day", port_metr_enabled=False)
    order = Order(
        stock_id="SH600000",
        amount=100.0,
        direction=OrderDir.BUY,
        start_time=pd.Timestamp("2020-01-02"),
        end_time=pd.Timestamp("2020-01-03"),
    )
    cash_before = account.get_cash()
    trade_val, trade_cost, trade_price = exchange.deal_order(order, trade_account=account)
    assert trade_price == pytest.approx(10.0)
    assert trade_val == pytest.approx(1000.0)
    assert trade_cost == pytest.approx(max(1000.0 * 0.0005, 5.0))
    assert order.deal_amount == pytest.approx(100.0)
    assert account.get_cash() == pytest.approx(cash_before - trade_val - trade_cost)
    assert account.current_position.get_stock_amount("SH600000") == pytest.approx(100.0)


def test_suspended_stock_is_not_tradable_and_deals_zero() -> None:
    frame = _synthetic_quote_df()
    frame.loc[("SH600000", pd.Timestamp("2020-01-02")), "$close"] = float("nan")
    provider = InMemoryQuoteProvider(quote_df=frame, codes=["SH600000"])
    exchange = Exchange(
        freq="day",
        start_time="2020-01-02",
        end_time="2020-01-06",
        codes=["SH600000"],
        deal_price="$close",
        limit_threshold=None,
        data_provider=provider,
    )
    assert not exchange.is_stock_tradable(
        "SH600000", pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-02"), OrderDir.BUY
    )
    account = Account(init_cash=1_000_000.0, position_dict={}, freq="day", port_metr_enabled=False)
    order = Order(
        stock_id="SH600000",
        amount=100.0,
        direction=OrderDir.BUY,
        start_time=pd.Timestamp("2020-01-02"),
        end_time=pd.Timestamp("2020-01-02"),
    )
    trade_val, trade_cost, _ = exchange.deal_order(order, trade_account=account)
    assert trade_val == pytest.approx(0.0)
    assert trade_cost == pytest.approx(0.0)
    assert order.deal_amount == pytest.approx(0.0)
    assert account.get_cash() == pytest.approx(1_000_000.0)

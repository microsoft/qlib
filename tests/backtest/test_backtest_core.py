"""Filesystem-free core backtest tests: order matching, account, validation, schemas."""

import logging

import pandas as pd
import pytest

from qlib.backtest.account import Account, AccumulatedInfo
from qlib.backtest.config_schema import ExchangeKwargsSchema, validate_benchmark_workflow
from qlib.backtest.decision import Order, OrderDir
from qlib.backtest.position import Position
from qlib.backtest.validation import (
    check_deal_price_expression,
    sanitize_cache_path,
    validate_order,
    validate_position_consistency,
)
from qlib.log_structured import JsonFormatter, get_json_logger, log_backtest_metrics


def _make_order(**overrides) -> Order:
    base = {
        "stock_id": "SH600000",
        "amount": 100.0,
        "direction": OrderDir.BUY,
        "start_time": pd.Timestamp("2020-01-02"),
        "end_time": pd.Timestamp("2020-01-03"),
    }
    base.update(overrides)
    return Order(**base)


def test_order_sign_and_amount_delta() -> None:
    buy = _make_order(direction=OrderDir.BUY, amount=10.0)
    sell = _make_order(direction=OrderDir.SELL, amount=10.0)
    assert buy.sign == 1
    assert sell.sign == -1
    assert buy.amount_delta == pytest.approx(10.0)
    assert sell.amount_delta == pytest.approx(-10.0)


def test_order_rejects_bad_direction() -> None:
    with pytest.raises(NotImplementedError):
        _make_order(direction=5)


def test_validate_order_bounds() -> None:
    validate_order(_make_order())
    with pytest.raises(ValueError):
        validate_order(_make_order(amount=-1.0))
    with pytest.raises(ValueError):
        validate_order(
            _make_order(start_time=pd.Timestamp("2020-01-05"), end_time=pd.Timestamp("2020-01-02"))
        )


def test_position_buy_then_sell_roundtrip() -> None:
    position = Position(cash=100000.0, position_dict={})
    buy = _make_order(direction=OrderDir.BUY, amount=100.0)
    position.update_order(buy, trade_val=1000.0, cost=5.0, trade_price=10.0)
    assert position.get_stock_amount("SH600000") == pytest.approx(100.0)
    assert position.get_cash() == pytest.approx(100000.0 - 1005.0)

    sell = _make_order(direction=OrderDir.SELL, amount=100.0)
    position.update_order(sell, trade_val=1200.0, cost=5.0, trade_price=12.0)
    assert position.get_stock_amount("SH600000") == pytest.approx(0.0)
    assert "SH600000" not in position.get_stock_list()
    validate_position_consistency(position)


def test_position_sell_missing_raises_key_error() -> None:
    position = Position(cash=1000.0, position_dict={})
    with pytest.raises(KeyError):
        position.update_order(
            _make_order(direction=OrderDir.SELL), trade_val=100.0, cost=0.0, trade_price=10.0
        )


def test_account_buy_updates_cash_turnover_and_cost() -> None:
    account = Account(
        init_cash=100000.0,
        position_dict={"SH600000": {"amount": 0, "price": 10.0}},
        freq="day",
        benchmark_config={},
        port_metr_enabled=False,
    )
    order = _make_order(direction=OrderDir.BUY, amount=100.0)
    account.update_order(order, trade_val=1000.0, cost=5.0, trade_price=10.0)
    assert account.get_cash() == pytest.approx(100000.0 - 1005.0)
    assert isinstance(account.accum_info, AccumulatedInfo)


def test_sanitize_cache_path_blocks_traversal(tmp_path) -> None:
    good = sanitize_cache_path(tmp_path, "features/SH600000.close.day.bin")
    assert str(good).startswith(str(tmp_path.resolve()))
    with pytest.raises(ValueError):
        sanitize_cache_path(tmp_path, "../../etc/passwd")


def test_check_deal_price_expression() -> None:
    assert check_deal_price_expression("close") == "$close"
    assert check_deal_price_expression("$vwap") == "$vwap"
    with pytest.raises(ValueError):
        check_deal_price_expression("import os; evil")


def test_exchange_kwargs_schema_defaults_and_validation() -> None:
    schema = ExchangeKwargsSchema.model_validate({"deal_price": "close", "open_cost": 0.0005})
    assert schema.deal_price == "$close"
    with pytest.raises(ValueError):
        ExchangeKwargsSchema.model_validate({"open_cost": -1.0})


def test_benchmark_workflow_schema_accepts_legacy_yaml() -> None:
    raw = {
        "qlib_init": {"provider_uri": "~/.qlib/qlib_data/cn_data", "region": "cn"},
        "market": "csi300",
        "benchmark": "SH000300",
        "port_analysis_config": {
            "strategy": {"class": "TopkDropoutStrategy"},
            "backtest": {
                "start_time": "2017-01-01",
                "end_time": "2020-08-01",
                "account": 100000000,
                "benchmark": "SH000300",
                "exchange_kwargs": {"deal_price": "close", "open_cost": 0.0005},
            },
        },
        "task": {},
    }
    parsed = validate_benchmark_workflow(raw)
    assert parsed.resolved_backtest.account == pytest.approx(100000000)
    assert parsed.resolved_backtest.exchange_kwargs.deal_price == "$close"


def test_json_logger_emits_structured_metrics(caplog) -> None:
    logger = get_json_logger("test_structured")
    assert isinstance(logger.handlers[0].formatter, JsonFormatter)
    with caplog.at_level(logging.INFO, logger="qlib.test_structured"):
        log_backtest_metrics(logger, sharpe=1.5, max_drawdown=-0.1, run_id="r1", strategy="topk")

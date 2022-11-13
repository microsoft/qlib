# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest

from qlib.backtest.decision import Order, OrderDir
from qlib.backtest.executor import SimulatorExecutor
from qlib.rl.order_execution import CategoricalActionInterpreter
from qlib.rl.order_execution.simulator_qlib import SingleAssetOrderExecution

TOTAL_POSITION = 2100.0

python_version_request = pytest.mark.skipif(sys.version_info < (3, 8), reason="requires python3.8 or higher")


def is_close(a: float, b: float, epsilon: float = 1e-4) -> bool:
    return abs(a - b) <= epsilon


def get_order() -> Order:
    return Order(
        stock_id="SH600000",
        amount=TOTAL_POSITION,
        direction=OrderDir.BUY,
        start_time=pd.Timestamp("2019-03-04 09:30:00"),
        end_time=pd.Timestamp("2019-03-04 14:29:00"),
    )


def get_configs(order: Order) -> Tuple[dict, dict]:
    executor_config = {
        "class": "NestedExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "1day",
            "inner_strategy": {"class": "ProxySAOEStrategy", "module_path": "qlib.rl.order_execution.strategy"},
            "track_data": True,
            "inner_executor": {
                "class": "NestedExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {
                    "time_per_step": "30min",
                    "inner_strategy": {
                        "class": "TWAPStrategy",
                        "module_path": "qlib.contrib.strategy.rule_strategy",
                    },
                    "inner_executor": {
                        "class": "SimulatorExecutor",
                        "module_path": "qlib.backtest.executor",
                        "kwargs": {
                            "time_per_step": "1min",
                            "verbose": False,
                            "trade_type": SimulatorExecutor.TT_SERIAL,
                            "generate_report": False,
                            "track_data": True,
                        },
                    },
                    "track_data": True,
                },
            },
            "start_time": pd.Timestamp(order.start_time.date()),
            "end_time": pd.Timestamp(order.start_time.date()),
        },
    }

    exchange_config = {
        "freq": "1min",
        "codes": [order.stock_id],
        "limit_threshold": ("$ask == 0", "$bid == 0"),
        "deal_price": ("If($ask == 0, $bid, $ask)", "If($bid == 0, $ask, $bid)"),
        "volume_threshold": {
            "all": ("cum", "0.2 * DayCumsum($volume, '9:30', '14:29')"),
            "buy": ("current", "$askV1"),
            "sell": ("current", "$bidV1"),
        },
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5.0,
        "trade_unit": None,
    }

    return executor_config, exchange_config


def get_simulator(order: Order) -> SingleAssetOrderExecution:
    DATA_ROOT_DIR = Path(__file__).parent.parent / ".data" / "rl" / "qlib_simulator"

    # fmt: off
    qlib_config = {
        "provider_uri_day": DATA_ROOT_DIR / "qlib_1d",
        "provider_uri_1min": DATA_ROOT_DIR / "qlib_1min",
        "feature_root_dir": DATA_ROOT_DIR / "qlib_handler_stock",
        "feature_columns_today": [
            "$open", "$high", "$low", "$close", "$vwap", "$bid", "$ask", "$volume",
            "$bidV", "$bidV1", "$bidV3", "$bidV5", "$askV", "$askV1", "$askV3", "$askV5",
        ],
        "feature_columns_yesterday": [
            "$open_1", "$high_1", "$low_1", "$close_1", "$vwap_1", "$bid_1", "$ask_1", "$volume_1",
            "$bidV_1", "$bidV1_1", "$bidV3_1", "$bidV5_1", "$askV_1", "$askV1_1", "$askV3_1", "$askV5_1",
        ],
    }
    # fmt: on

    executor_config, exchange_config = get_configs(order)

    return SingleAssetOrderExecution(
        order=order,
        qlib_config=qlib_config,
        executor_config=executor_config,
        exchange_config=exchange_config,
    )


@python_version_request
def test_simulator_first_step():
    order = get_order()
    simulator = get_simulator(order)
    state = simulator.get_state()
    assert state.cur_time == pd.Timestamp("2019-03-04 09:30:00")
    assert state.position == TOTAL_POSITION

    AMOUNT = 300.0
    simulator.step(AMOUNT)
    state = simulator.get_state()
    assert state.cur_time == pd.Timestamp("2019-03-04 10:00:00")
    assert state.position == TOTAL_POSITION - AMOUNT
    assert len(state.history_exec) == 30
    assert state.history_exec.index[0] == pd.Timestamp("2019-03-04 09:30:00")

    assert is_close(state.history_exec["market_volume"].iloc[0], 109382.382812)
    assert is_close(state.history_exec["market_price"].iloc[0], 149.566483)
    assert (state.history_exec["amount"] == AMOUNT / 30).all()
    assert (state.history_exec["deal_amount"] == AMOUNT / 30).all()
    assert is_close(state.history_exec["trade_price"].iloc[0], 149.566483)
    assert is_close(state.history_exec["trade_value"].iloc[0], 1495.664825)
    assert is_close(state.history_exec["position"].iloc[0], TOTAL_POSITION - AMOUNT / 30)
    assert is_close(state.history_exec["ffr"].iloc[0], AMOUNT / TOTAL_POSITION / 30)

    assert is_close(state.history_steps["market_volume"].iloc[0], 1254848.5756835938)
    assert state.history_steps["amount"].iloc[0] == AMOUNT
    assert state.history_steps["deal_amount"].iloc[0] == AMOUNT
    assert state.history_steps["ffr"].iloc[0] == AMOUNT / TOTAL_POSITION
    assert is_close(
        state.history_steps["pa"].iloc[0] * (1.0 if order.direction == OrderDir.SELL else -1.0),
        (state.history_steps["trade_price"].iloc[0] / simulator.twap_price - 1) * 10000,
    )


@python_version_request
def test_simulator_stop_twap() -> None:
    order = get_order()
    simulator = get_simulator(order)
    NUM_STEPS = 7
    for i in range(NUM_STEPS):
        simulator.step(TOTAL_POSITION / NUM_STEPS)

    HISTORY_STEP_LENGTH = 30 * NUM_STEPS
    state = simulator.get_state()
    assert len(state.history_exec) == HISTORY_STEP_LENGTH

    assert (state.history_exec["deal_amount"] == TOTAL_POSITION / HISTORY_STEP_LENGTH).all()
    assert is_close(state.history_steps["position"].iloc[0], TOTAL_POSITION * (NUM_STEPS - 1) / NUM_STEPS)
    assert is_close(state.history_steps["position"].iloc[-1], 0.0)
    assert is_close(state.position, 0.0)
    assert is_close(state.metrics["ffr"], 1.0)

    assert is_close(state.metrics["market_price"], state.backtest_data.get_deal_price().mean())
    assert is_close(state.metrics["market_volume"], state.backtest_data.get_volume().sum())
    assert is_close(state.metrics["trade_price"], state.metrics["market_price"])
    assert is_close(state.metrics["pa"], 0.0)

    assert simulator.done()


@python_version_request
def test_interpreter() -> None:
    NUM_EXECUTION = 3
    order = get_order()
    simulator = get_simulator(order)
    interpreter_action = CategoricalActionInterpreter(values=NUM_EXECUTION)

    NUM_STEPS = 7
    state = simulator.get_state()
    position_history = []
    for i in range(NUM_STEPS):
        simulator.step(interpreter_action(state, 1))
        state = simulator.get_state()
        position_history.append(state.position)

        assert position_history[-1] == max(TOTAL_POSITION - TOTAL_POSITION / NUM_EXECUTION * (i + 1), 0.0)

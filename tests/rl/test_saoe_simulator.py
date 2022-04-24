# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path

import pandas as pd
import pytest

from qlib.backtest import Order
from qlib.rl.data import pickle_styled
from qlib.rl.order_execution import SingleAssetOrderExecution


DATA_DIR = Path("/mnt/data/Sample-Testdata/us/")  # Update this when infrastructure is built.
BACKTEST_DATA_DIR = DATA_DIR / "raw"


def test_pickle_data_inspect():
    data = pickle_styled.get_intraday_backtest_data(BACKTEST_DATA_DIR, "AAL", "2013-12-11", "close", 0)
    assert len(data) == 390

    data = pickle_styled.get_intraday_processed_data(DATA_DIR / "processed", "AAL", "2013-12-11", 5, data.get_time_index())
    assert len(data.today) == len(data.yesterday) == 390


def test_simulator_first_step():
    order = Order(
        "AAL", 30., 0,
        pd.Timestamp("2013-12-11 00:00:00"),
        pd.Timestamp("2013-12-11 23:59:59")
    )

    simulator = SingleAssetOrderExecution(order, BACKTEST_DATA_DIR)
    state = simulator.get_state()
    assert state.cur_time == pd.Timestamp("2013-12-11 09:30:00")
    assert state.position == 30.

    simulator.step(15.)
    state = simulator.get_state()
    assert len(state.history_exec) == 30
    assert state.history_exec.index[0] == pd.Timestamp("2013-12-11 09:30:00")
    assert state.history_exec["market_volume"].iloc[0] == 450072.0
    assert abs(state.history_exec["market_price"].iloc[0] - 25.370001) < 1e-4
    assert (state.history_exec["amount"] == 0.5).all()
    assert (state.history_exec["deal_amount"] == 0.5).all()
    assert abs(state.history_exec["trade_price"].iloc[0] - 25.370001) < 1e-4
    assert abs(state.history_exec["trade_value"].iloc[0] - 12.68500) < 1e-4
    assert state.history_exec["position"].iloc[0] == 29.5
    assert state.history_exec["ffr"].iloc[0] == 1 / 60

    assert state.history_steps["market_volume"].iloc[0] == 5041147.0
    assert state.history_steps["amount"].iloc[0] == 15.
    assert state.history_steps["deal_amount"].iloc[0] == 15.
    assert state.history_steps["ffr"].iloc[0] == 0.5
    assert state.history_steps["pa"].iloc[0] == (state.history_steps["trade_price"].iloc[0] / simulator.twap_price - 1) * 10000

    assert state.position == 15.
    assert state.cur_time == pd.Timestamp("2013-12-11 10:00:00")


def test_simulator_stop_twap():
    order = Order(
        "AAL", 13., 0,
        pd.Timestamp("2013-12-11 00:00:00"),
        pd.Timestamp("2013-12-11 23:59:59")
    )

    simulator = SingleAssetOrderExecution(order, BACKTEST_DATA_DIR)
    for _ in range(13):
        simulator.step(1.)

    state = simulator.get_state()
    assert len(state.history_exec) == 390
    assert (state.history_exec["deal_amount"] == 13 / 390).all()
    assert state.history_steps["position"].iloc[0] == 12 and state.history_steps["position"].iloc[-1] == 0

    assert (state.metrics["ffr"] - 1) < 1e-3
    assert state.metrics["market_price"] == state.backtest_data.get_deal_price().mean()
    assert state.metrics["market_volume"] == state.backtest_data.get_volume().sum()
    assert state.position == 0.
    assert abs(state.metrics["trade_price"] - state.metrics["market_price"]) < 1e-4
    assert abs(state.metrics["pa"]) < 1e-2

    assert simulator.done()


def test_simulator_stop_early():
    order = Order("AAL", 1., 1, 
                  pd.Timestamp("2013-12-11 00:00:00"),
                  pd.Timestamp("2013-12-11 23:59:59"))

    with pytest.raises(ValueError):
        simulator = SingleAssetOrderExecution(order, BACKTEST_DATA_DIR)
        simulator.step(2.)

    simulator = SingleAssetOrderExecution(order, BACKTEST_DATA_DIR)
    simulator.step(1.)

    with pytest.raises(AssertionError):
        simulator.step(1.)

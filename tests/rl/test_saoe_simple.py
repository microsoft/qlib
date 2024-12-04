# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
import torch
from tianshou.data import Batch

from qlib.backtest import Order
from qlib.config import C
from qlib.log import set_log_with_config
from qlib.rl.data import pickle_styled
from qlib.rl.data.pickle_styled import PickleProcessedDataProvider
from qlib.rl.order_execution import *
from qlib.rl.trainer import backtest, train
from qlib.rl.utils import ConsoleWriter, CsvWriter, EnvWrapperStatus

pytestmark = pytest.mark.skipif(sys.version_info < (3, 8), reason="Pickle styled data only supports Python >= 3.8")


DATA_ROOT_DIR = Path(__file__).parent.parent / ".data" / "rl" / "intraday_saoe"
DATA_DIR = DATA_ROOT_DIR / "us"
BACKTEST_DATA_DIR = DATA_DIR / "backtest"
FEATURE_DATA_DIR = DATA_DIR / "processed"
ORDER_DIR = DATA_DIR / "order" / "valid_bidir"

CN_DATA_DIR = DATA_ROOT_DIR / "cn"
CN_FEATURE_DATA_DIR = CN_DATA_DIR / "processed"
CN_ORDER_DIR = CN_DATA_DIR / "order" / "test"
CN_POLICY_WEIGHTS_DIR = CN_DATA_DIR / "weights"


def test_pickle_data_inspect():
    data = pickle_styled.load_simple_intraday_backtest_data(BACKTEST_DATA_DIR, "AAL", "2013-12-11", "close", 0)
    assert len(data) == 390

    provider = PickleProcessedDataProvider(DATA_DIR / "processed")
    data = provider.get_data("AAL", "2013-12-11", 5, data.get_time_index())
    assert len(data.today) == len(data.yesterday) == 390


def test_simulator_first_step():
    order = Order("AAL", 30.0, 0, pd.Timestamp("2013-12-11 00:00:00"), pd.Timestamp("2013-12-11 23:59:59"))

    simulator = SingleAssetOrderExecutionSimple(order, DATA_DIR)
    state = simulator.get_state()
    assert state.cur_time == pd.Timestamp("2013-12-11 09:30:00")
    assert state.position == 30.0

    simulator.step(15.0)
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
    assert state.history_steps["amount"].iloc[0] == 15.0
    assert state.history_steps["deal_amount"].iloc[0] == 15.0
    assert state.history_steps["ffr"].iloc[0] == 0.5
    assert (
        state.history_steps["pa"].iloc[0]
        == (state.history_steps["trade_price"].iloc[0] / simulator.twap_price - 1) * 10000
    )

    assert state.position == 15.0
    assert state.cur_time == pd.Timestamp("2013-12-11 10:00:00")


def test_simulator_stop_twap():
    order = Order("AAL", 13.0, 0, pd.Timestamp("2013-12-11 00:00:00"), pd.Timestamp("2013-12-11 23:59:59"))

    simulator = SingleAssetOrderExecutionSimple(order, DATA_DIR)
    for _ in range(13):
        simulator.step(1.0)

    state = simulator.get_state()
    assert len(state.history_exec) == 390
    assert (state.history_exec["deal_amount"] == 13 / 390).all()
    assert state.history_steps["position"].iloc[0] == 12 and state.history_steps["position"].iloc[-1] == 0

    assert (state.metrics["ffr"] - 1) < 1e-3
    assert abs(state.metrics["market_price"] - state.backtest_data.get_deal_price().mean()) < 1e-4
    assert np.isclose(state.metrics["market_volume"], state.backtest_data.get_volume().sum())
    assert state.position == 0.0
    assert abs(state.metrics["trade_price"] - state.metrics["market_price"]) < 1e-4
    assert abs(state.metrics["pa"]) < 1e-2

    assert simulator.done()


def test_simulator_stop_early():
    order = Order("AAL", 1.0, 1, pd.Timestamp("2013-12-11 00:00:00"), pd.Timestamp("2013-12-11 23:59:59"))

    with pytest.raises(ValueError):
        simulator = SingleAssetOrderExecutionSimple(order, DATA_DIR)
        simulator.step(2.0)

    simulator = SingleAssetOrderExecutionSimple(order, DATA_DIR)
    simulator.step(1.0)

    with pytest.raises(AssertionError):
        simulator.step(1.0)


def test_simulator_start_middle():
    order = Order("AAL", 15.0, 1, pd.Timestamp("2013-12-11 10:15:00"), pd.Timestamp("2013-12-11 15:44:59"))

    simulator = SingleAssetOrderExecutionSimple(order, DATA_DIR)
    assert len(simulator.ticks_for_order) == 330
    assert simulator.cur_time == pd.Timestamp("2013-12-11 10:15:00")
    simulator.step(2.0)
    assert simulator.cur_time == pd.Timestamp("2013-12-11 10:30:00")

    for _ in range(10):
        simulator.step(1.0)

    simulator.step(2.0)
    assert len(simulator.history_exec) == 330
    assert simulator.done()
    assert abs(simulator.history_exec["amount"].iloc[-1] - (1 + 2 / 15)) < 1e-4
    assert abs(simulator.metrics["ffr"] - 1) < 1e-4


def test_interpreter():
    order = Order("AAL", 15.0, 1, pd.Timestamp("2013-12-11 10:15:00"), pd.Timestamp("2013-12-11 15:44:59"))

    simulator = SingleAssetOrderExecutionSimple(order, DATA_DIR)
    assert len(simulator.ticks_for_order) == 330
    assert simulator.cur_time == pd.Timestamp("2013-12-11 10:15:00")

    # emulate a env status
    class EmulateEnvWrapper(NamedTuple):
        status: EnvWrapperStatus

    interpreter = FullHistoryStateInterpreter(13, 390, 5, PickleProcessedDataProvider(FEATURE_DATA_DIR))
    interpreter_step = CurrentStepStateInterpreter(13)
    interpreter_action = CategoricalActionInterpreter(20)
    interpreter_action_twap = TwapRelativeActionInterpreter()

    wrapper_status_kwargs = dict(initial_state=order, obs_history=[], action_history=[], reward_history=[])

    # first step
    interpreter.env = EmulateEnvWrapper(status=EnvWrapperStatus(cur_step=0, done=False, **wrapper_status_kwargs))

    obs = interpreter(simulator.get_state())
    assert obs["cur_tick"] == 45
    assert obs["cur_step"] == 0
    assert obs["position"] == 15.0
    assert obs["position_history"][0] == 15.0
    assert all(np.sum(obs["data_processed"][i]) != 0 for i in range(45))
    assert np.sum(obs["data_processed"][45:]) == 0
    assert obs["data_processed_prev"].shape == (390, 5)

    # first step: second interpreter
    interpreter_step.env = EmulateEnvWrapper(status=EnvWrapperStatus(cur_step=0, done=False, **wrapper_status_kwargs))

    obs = interpreter_step(simulator.get_state())
    assert obs["acquiring"] == 1
    assert obs["position"] == 15.0

    # second step
    simulator.step(5.0)
    interpreter.env = EmulateEnvWrapper(status=EnvWrapperStatus(cur_step=1, done=False, **wrapper_status_kwargs))

    obs = interpreter(simulator.get_state())
    assert obs["cur_tick"] == 60
    assert obs["cur_step"] == 1
    assert obs["position"] == 10.0
    assert obs["position_history"][:2].tolist() == [15.0, 10.0]
    assert all(np.sum(obs["data_processed"][i]) != 0 for i in range(60))
    assert np.sum(obs["data_processed"][60:]) == 0

    # second step: action
    action = interpreter_action(simulator.get_state(), 1)
    assert action == 15 / 20

    interpreter_action_twap.env = EmulateEnvWrapper(
        status=EnvWrapperStatus(cur_step=1, done=False, **wrapper_status_kwargs)
    )
    action = interpreter_action_twap(simulator.get_state(), 1.5)
    assert action == 1.5

    # fast-forward
    for _ in range(10):
        simulator.step(0.0)

    # last step
    simulator.step(5.0)
    interpreter.env = EmulateEnvWrapper(
        status=EnvWrapperStatus(cur_step=12, done=simulator.done(), **wrapper_status_kwargs)
    )

    assert interpreter.env.status["done"]

    obs = interpreter(simulator.get_state())
    assert obs["cur_tick"] == 375
    assert obs["cur_step"] == 12
    assert obs["position"] == 0.0
    assert obs["position_history"][1:11].tolist() == [10.0] * 10
    assert all(np.sum(obs["data_processed"][i]) != 0 for i in range(375))
    assert np.sum(obs["data_processed"][375:]) == 0


def test_network_sanity():
    # we won't check the correctness of networks here
    order = Order("AAL", 15.0, 1, pd.Timestamp("2013-12-11 9:30:00"), pd.Timestamp("2013-12-11 15:59:59"))

    simulator = SingleAssetOrderExecutionSimple(order, DATA_DIR)
    assert len(simulator.ticks_for_order) == 390

    class EmulateEnvWrapper(NamedTuple):
        status: EnvWrapperStatus

    interpreter = FullHistoryStateInterpreter(13, 390, 5, PickleProcessedDataProvider(FEATURE_DATA_DIR))
    action_interp = CategoricalActionInterpreter(13)

    wrapper_status_kwargs = dict(initial_state=order, obs_history=[], action_history=[], reward_history=[])

    network = Recurrent(interpreter.observation_space)
    policy = PPO(network, interpreter.observation_space, action_interp.action_space, 1e-3)

    for i in range(14):
        interpreter.env = EmulateEnvWrapper(status=EnvWrapperStatus(cur_step=i, done=False, **wrapper_status_kwargs))
        obs = interpreter(simulator.get_state())
        batch = Batch(obs=[obs])
        output = policy(batch)
        assert 0 <= output["act"].item() <= 13
        if i < 13:
            simulator.step(1.0)
        else:
            assert obs["cur_tick"] == 389
            assert obs["cur_step"] == 12
            assert obs["position_history"][-1] == 3


@pytest.mark.parametrize("finite_env_type", ["dummy", "subproc", "shmem"])
def test_twap_strategy(finite_env_type):
    set_log_with_config(C.logging_config)
    orders = pickle_styled.load_orders(ORDER_DIR)
    assert len(orders) == 248

    state_interp = FullHistoryStateInterpreter(13, 390, 5, PickleProcessedDataProvider(FEATURE_DATA_DIR))
    action_interp = TwapRelativeActionInterpreter()
    policy = AllOne(state_interp.observation_space, action_interp.action_space)
    csv_writer = CsvWriter(Path(__file__).parent / ".output")

    backtest(
        partial(SingleAssetOrderExecutionSimple, data_dir=DATA_DIR, ticks_per_step=30),
        state_interp,
        action_interp,
        orders,
        policy,
        [ConsoleWriter(total_episodes=len(orders)), csv_writer],
        concurrency=4,
        finite_env_type=finite_env_type,
    )

    metrics = pd.read_csv(Path(__file__).parent / ".output" / "result.csv")
    assert len(metrics) == 248
    assert np.isclose(metrics["ffr"].mean(), 1.0)
    assert np.isclose(metrics["pa"].mean(), 0.0)
    assert np.allclose(metrics["pa"], 0.0, atol=2e-3)


def test_cn_ppo_strategy():
    set_log_with_config(C.logging_config)
    # The data starts with 9:31 and ends with 15:00
    orders = pickle_styled.load_orders(CN_ORDER_DIR, start_time=pd.Timestamp("9:31"), end_time=pd.Timestamp("14:58"))
    assert len(orders) == 40

    state_interp = FullHistoryStateInterpreter(8, 240, 6, PickleProcessedDataProvider(CN_FEATURE_DATA_DIR))
    action_interp = CategoricalActionInterpreter(4)
    network = Recurrent(state_interp.observation_space)
    policy = PPO(network, state_interp.observation_space, action_interp.action_space, 1e-4)
    policy.load_state_dict(torch.load(CN_POLICY_WEIGHTS_DIR / "ppo_recurrent_30min.pth", map_location="cpu"))
    csv_writer = CsvWriter(Path(__file__).parent / ".output")

    backtest(
        partial(SingleAssetOrderExecutionSimple, data_dir=CN_DATA_DIR, ticks_per_step=30),
        state_interp,
        action_interp,
        orders,
        policy,
        [ConsoleWriter(total_episodes=len(orders)), csv_writer],
        concurrency=4,
    )

    metrics = pd.read_csv(Path(__file__).parent / ".output" / "result.csv")
    assert len(metrics) == len(orders)
    assert np.isclose(metrics["ffr"].mean(), 1.0)
    assert np.isclose(metrics["pa"].mean(), -16.21578303474833)
    assert np.isclose(metrics["market_price"].mean(), 58.68277690875527)
    assert np.isclose(metrics["trade_price"].mean(), 58.76063985000002)


def test_ppo_train():
    set_log_with_config(C.logging_config)
    # The data starts with 9:31 and ends with 15:00
    orders = pickle_styled.load_orders(CN_ORDER_DIR, start_time=pd.Timestamp("9:31"), end_time=pd.Timestamp("14:58"))
    assert len(orders) == 40

    state_interp = FullHistoryStateInterpreter(8, 240, 6, PickleProcessedDataProvider(CN_FEATURE_DATA_DIR))
    action_interp = CategoricalActionInterpreter(4)
    network = Recurrent(state_interp.observation_space)
    policy = PPO(network, state_interp.observation_space, action_interp.action_space, 1e-4)

    train(
        partial(SingleAssetOrderExecutionSimple, data_dir=CN_DATA_DIR, ticks_per_step=30),
        state_interp,
        action_interp,
        orders,
        policy,
        PAPenaltyReward(),
        vessel_kwargs={"episode_per_iter": 100, "update_kwargs": {"batch_size": 64, "repeat": 5}},
        trainer_kwargs={"max_iters": 2, "loggers": ConsoleWriter(total_episodes=100)},
    )

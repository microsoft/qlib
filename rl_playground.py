import logging
import pickle
from enum import Enum
from typing import Iterable, Optional, Any

import gym
import numpy as np

import torch
from torch.utils.data import Dataset

from qlib.backtest import get_exchange, Account, BaseExecutor
from qlib.rl.interpreter import StateInterpreter, ActionInterpreter
from qlib.utils import init_instance_by_config


def get_executor(start_time, end_time, executor, benchmark="SH000300", account=1e9, exchange_kwargs={}):
    trade_account = Account(
        init_cash=account,
        benchmark_config={
            "benchmark": benchmark,
            "start_time": start_time,
            "end_time": end_time,
        },
    )
    trade_exchange = get_exchange(**exchange_kwargs)

    common_infra = {
        "trade_account": trade_account,
        "trade_exchange": trade_exchange,
    }

    trade_executor = init_instance_by_config(executor, accept_types=BaseExecutor, common_infra=common_infra)

    return common_infra, trade_executor


class QlibOrderDataset(Dataset):
    def __init__(self, order_file):
        with open(order_file, 'rb') as f:
            self.orders = pickle.load(f)

    def __len__(self):
        return len(self.orders)

    def __getitem__(self, index):
        return self.orders[index]


class OrderEnv(gym.Env):
    def __init__(self,
                 state_interpreter: StateInterpreter,
                 action_interpreter: ActionInterpreter,
                 reward: Any,
                 dataloader: Iterable,
                 executor: BaseExecutor):
        self.action_interpreter = action_interpreter
        self.state_interpreter = state_interpreter
        self.reward = reward
        self.dataloader = dataloader
        self.executor = executor

    @property
    def action_space(self):
        return self.action.action_space

    @property
    def observation_space(self):
        return self.observation.observation_space

    def reset(self):
        try:
            self.cur_order = next(self.dataloader)
        except StopIteration:
            self.dataloader = None
            return None

        self.executor.reset(start_time=self.cur_order.start_time, end_time=self.cur_order.end_time)
        self.level_infra = self.executor.get_level_infra()
        self.execute_result = []

        # TODO: how to fetch data after feature engineering?

        # TODO: can be rewritten as dataclasses.asdict(self.cur_order) is Order is written to be a dataclass
        return self.state_interpreter(self.cur_order, self.level_infra)

    def step(self, action):
        assert self.dataloader is not None

        assert not self.executor.finished()

        trade_decision = self.action_interpreter(action)
        self.execute_result.extend(self.executor.execute(trade_decision))
        reward, rew_info = self.reward()

        done = self.executor.finished()
        info = {
            'action_history': self.action_history,
            'category': self.ep_state.flow_dir.value,
            'reward': rew_info
        }
        if self.ep_state.done:
            info['logs'] = self.ep_state.logs()
            info['index'] = {
                'ins': self._sample.ins,
                'date': self._sample.date
            }

        # TODO: how to collect metrics
        return self.state_interpreter(self.cur_order, self.level_infra), reward, done, info


def _main():
    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "verbose": True,
            "generate_report": True,
        }
    }
    # TODO: why is there a benchmark?
    trade_start_time = "2017-01-01"
    trade_end_time = "2020-08-01"
    benchmark = "SH000300"
    executor = get_executor(
        trade_start_time, trade_end_time, executor_config,
        benchmark, 1000000000, exchange_kwargs={
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        }
    )

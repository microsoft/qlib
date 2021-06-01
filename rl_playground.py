import pickle
from dataclasses import dataclass
from typing import Iterable, Any

import numpy as np
import gym
import qlib
from qlib.backtest import get_exchange, Account, BaseExecutor, CommonInfrastructure, Order
from qlib.config import REG_CN
from qlib.data import D
from qlib.rl.interpreter import StateInterpreter, ActionInterpreter
from qlib.tests.data import GetData
from qlib.utils import init_instance_by_config, exists_qlib_data
from torch.utils.data import Dataset, DataLoader
from tianshou.data import Batch, Collector
from tianshou.env import DummyVectorEnv
from tianshou.policy import BasePolicy


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

    common_infra = CommonInfrastructure(trade_account=trade_account, trade_exchange=trade_exchange)
    trade_executor = init_instance_by_config(executor, accept_types=BaseExecutor, common_infra=common_infra)

    return trade_executor


class QlibOrderDataset(Dataset):
    def __init__(self, order_file):
        with open(order_file, 'rb') as f:
            self.orders = pickle.load(f)

    def __len__(self):
        return len(self.orders)

    def __getitem__(self, index):
        return self.orders[index]


class DummyCallable:
    def __call__(self, *args, **kwargs):
        if args:
            return args[0]
        if kwargs:
            for v in kwargs.values():
                return v


class DummyPolicy(BasePolicy):
    def forward(self, batch, state=None, **kwargs):
        return Batch(act=0)

    def learn(self, *args, **kwargs):
        pass


@dataclass
class EpisodicState:
    """
    A simplified data structure for RL-related components to process observations and rewards
    """
    # requirements
    start_time: int
    end_time: int
    num_step: int
    time_per_step: int
    target: float
    target_limit: float
    vol_limit: Optional[float]
    flow_dir: int
    market_price: np.ndarray
    market_vol: np.ndarray

    # agent state
    cur_time: int = -1
    cur_step: int = 0
    done: bool = False
    position: Optional[float] = None
    exec_vol: Optional[np.ndarray] = None
    last_step_duration: Optional[int] = None
    position_history: Optional[np.ndarray] = None

    # calculated statistics
    turnover: Optional[float] = None
    baseline_twap: Optional[float] = None
    baseline_vwap: Optional[float] = None
    exec_avg_price: Optional[float] = None
    pa_twap: Optional[float] = None
    pa_vwap: Optional[float] = None
    fulfill_rate: Optional[float] = None

    def __post_init__(self):
        assert self.target >= 0
        self.cur_time = self.start_time
        self.position = self.target
        self.position_history = np.full((self.num_step + 1), np.nan)
        self.position_history[0] = self.position
        self.baseline_twap = np.mean(self.market_price)
        if self.market_vol.sum() == 0:
            self.baseline_vwap = np.mean(self.market_price)
        else:
            self.baseline_vwap = np.average(self.market_price, weights=self.market_vol)

    def update_stats(self):
        market_price = self.market_price[:len(self.exec_vol)]
        self.turnover = (self.exec_vol * market_price).sum()
        # exec_vol can be zero
        if np.isclose(self.exec_vol.sum(), 0):
            self.exec_avg_price = market_price[0]
        else:
            self.exec_avg_price = np.average(market_price, weights=self.exec_vol)
        self.pa_twap = price_advantage(self.exec_avg_price, self.baseline_twap, self.flow_dir)
        self.pa_vwap = price_advantage(self.exec_avg_price, self.baseline_vwap, self.flow_dir)
        self.fulfill_rate = (self.target - self.position) / self.target_limit
        if abs(self.fulfill_rate - 1.0) < EPSILON:
            self.fulfill_rate = 1.0
        self.fulfill_rate *= 100

    def logs(self):
        logs = {
            'stop_time': self.cur_time - self.start_time,
            'stop_step': self.cur_step,
            'turnover': self.turnover,
            'baseline_twap': self.baseline_twap,
            'baseline_vwap': self.baseline_vwap,
            'exec_avg_price': self.exec_avg_price,
            'pa_twap': self.pa_twap,
            'pa_vwap': self.pa_vwap,
            'ffr': self.fulfill_rate
        }
        return logs

    def next_duration(self) -> int:
        return min(self.time_per_step, self.end_time - self.cur_time)

    def step(self, exec_vol):
        self.last_step_duration = len(exec_vol)
        self.position -= exec_vol.sum()
        assert self.position > -EPSILON and (exec_vol > -EPSILON).all(), \
            f'Execution volume is invalid: {exec_vol} (position = {self.position})'
        self.position_history[self.cur_step + 1] = self.position
        self.cur_time += self.last_step_duration
        self.cur_step += 1
        if self.cur_step == self.num_step:
            assert self.cur_time == self.end_time
        if self.exec_vol is None:
            self.exec_vol = exec_vol
        else:
            self.exec_vol = np.concatenate((self.exec_vol, exec_vol))

        self.done = self.position < EPSILON or self.cur_step == self.num_step
        if self.done:
            self.update_stats()

        l, r = self.cur_time - self.last_step_duration - self.start_time, self.cur_time - self.start_time
        assert 0 <= l < r
        return StepState(self.exec_vol[l:r], self.market_vol[l:r], self.market_price[l:r], self)


@dataclass
class StepState:
    exec_vol: np.ndarray
    market_vol: np.ndarray
    market_price: np.ndarray

    # episode info
    episode_state: EpisodicState

    # calculated statistics
    turnover: Optional[float] = None
    exec_avg_price: Optional[float] = None
    pa_twap: Optional[float] = None
    pa_vwap: Optional[float] = None

    def __post_init__(self):
        assert len(self.exec_vol) == len(self.market_price) == len(self.market_vol)
        self.turnover = (self.exec_vol * self.market_price).sum()
        if np.isclose(self.market_vol.sum(), 0):
            self.exec_avg_price = self.market_price[0]
        else:
            self.exec_avg_price = np.average(self.market_price, weights=self.market_vol)
        self.pa_twap = price_advantage(self.exec_avg_price, self.episode_state.baseline_twap,
                                       self.episode_state.flow_dir)
        self.pa_vwap = price_advantage(self.exec_avg_price, self.episode_state.baseline_vwap,
                                       self.episode_state.flow_dir)


def price_advantage(exec_price: float, baseline_price: float, flow: FlowDirection) -> float:
    if baseline_price == 0:
        return 0.
    if flow == FlowDirection.ACQUIRE:
        return (1 - exec_price / baseline_price) * 10000
    else:
        return (exec_price / baseline_price - 1) * 10000



class SingleOrderEnv(gym.Env):
    MAX_STEPS = 10
    def __init__(self,
                 observation: StateInterpreter,
                 action: ActionInterpreter,
                 reward: Any,
                 dataloader: Iterable,
                 executor: BaseExecutor):
        self.action = action
        self.observation = observation
        self.reward = reward
        self.dataloader = dataloader
        self.executor = executor

        self.inner_frequency = self.executor.get_all_executor()[-1].time_per_step

    @property
    def action_space(self):
        return self.action.action_space

    @property
    def observation_space(self):
        return self.observation.observation_space

    def retrieve_data(self, cur_order: Order):
        return D.features(
            [cur_order.stock_id],
            ['$open', '$close', '$high', '$low', '$volume'],
            start_time=cur_order.start_time.date(),
            end_time=cur_order.end_time.date(),
            freq=self.inner_frequency
        )

    def initialize_state(self):
        self.executor.reset(start_time=self.cur_order.start_time, end_time=self.cur_order.end_time)
        return EpisodicState()

    def update_state(self, action):
        trade_decision = action
        execute_result = self.executor.execute(trade_decision)

    def reset(self):
        try:
            cur_order = next(self.dataloader)
        except StopIteration:
            self.dataloader = None
            return None

        self.cur_sample = self._retrieve_data(cur_order)
        self.execute_result = []
        self.ep_state = self.initialize_state()

        self.action_history = np.full(self.MAX_STEPS, np.nan)
        return self.observation(self.cur_sample, self.ep_state)


        # TODO: how to fetch data after feature engineering?

        # TODO: can be rewritten as dataclasses.asdict(self.cur_order) is Order is written to be a dataclass
        return self.observation

    def step(self, action):
        assert self.dataloader is not None

        assert not self.executor.finished()

        exec_vol = self.action(action, self.ep_state)
        step_state = self.ep_state.step(exec_vol)

        reward, rew_info = self.reward(self.ep_state, step_state)

        info = {
            'action_history': self.action_history,
            'category': self.ep_state.flow_dir.value,
            'reward': rew_info
        }
        if self.ep_state.done:
            info['logs'] = self.ep_state.logs()
            info['index'] = {
                'ins': self.cur_sample.ins,
                'date': self.cur_sample.date
            }

        return self.observation(self.cur_sample, self.ep_state), reward, self.ep_state.done, info


def _init_qlib():
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    if not exists_qlib_data(provider_uri):
        print(f"Qlib data is not found in {provider_uri}")
        GetData().qlib_data(target_dir=provider_uri, region=REG_CN)
    qlib.init(provider_uri=provider_uri, region=REG_CN)


def _main():
    _init_qlib()

    # TODO: why is there a benchmark?
    trade_start_time = "2017-01-01"
    trade_end_time = "2020-08-01"
    benchmark = "SH000300"
    time_per_step = "day"
    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": time_per_step,
            "verbose": True,
            "generate_report": False,
        }
    }
    executor = get_executor(
        trade_start_time,
        trade_end_time,
        executor_config,
        benchmark,
        1000000000,
        exchange_kwargs={
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        }
    )

    import pdb; pdb.set_trace()

    observation = DummyCallable()
    action = DummyCallable()
    reward_fn = DummyCallable()
    # TODO: this probably won't work with multiprocess
    dataloader = iter(DataLoader(QlibOrderDataset('rl.pkl'), batch_size=None, shuffle=True))

    def dummy_env(): return OrderEnv(observation, action, reward_fn, dataloader, executor)
    policy = DummyPolicy()

    # env = dummy_env()
    # obs = env.reset()
    # print(obs.__dict__)

    envs = DummyVectorEnv([dummy_env for _ in range(4)])
    test_collector = Collector(policy, envs)
    policy.eval()
    test_collector.collect(n_episode=10)


if __name__ == '__main__':
    _main()

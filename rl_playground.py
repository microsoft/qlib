import pickle
from dataclasses import dataclass, asdict
from pprint import pprint
from typing import Iterable, Any, Optional, Tuple, Dict

import gym
import numpy as np
import pandas as pd
import qlib
from gym import spaces
from qlib.backtest import get_exchange, Account, BaseExecutor, CommonInfrastructure, Order
from qlib.config import REG_CN
from qlib.data import D
from qlib.rl.interpreter import StateInterpreter, ActionInterpreter
from qlib.tests.data import GetData
from qlib.utils import init_instance_by_config, exists_qlib_data
from torch.utils.data import Dataset, DataLoader
from tianshou.data import Batch, Collector
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import BasePolicy


MAX_STEPS = 10


def get_executor(start_time, end_time, executor, exchange, benchmark="SH000300", account=1e9) -> BaseExecutor:
    trade_account = Account(
        init_cash=account,
        benchmark_config={
            "benchmark": benchmark,
            "start_time": start_time,
            "end_time": end_time,
        },
    )

    common_infra = CommonInfrastructure(trade_account=trade_account, trade_exchange=exchange)
    trade_executor = init_instance_by_config(executor, accept_types=BaseExecutor, common_infra=common_infra)

    return trade_executor


def price_advantage(exec_price: float, baseline_price: float, direction: int) -> float:
    if baseline_price == 0:
        return 0.
    if direction == 1:
        return (1 - exec_price / baseline_price) * 10000
    else:
        return (exec_price / baseline_price - 1) * 10000


@dataclass
class EpisodicState:
    """
    A simplified data structure as the input of RL-related components to calculate observations and rewards.
    Some of the metrics info are calculated on-the-fly in this class.
    """
    # requirements
    stock_id: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    direction: int
    target: float
    num_step: int

    # simplified market data used to calculate backtest metrics
    # this may contains information from future so be careful
    market_price: np.ndarray
    market_vol: np.ndarray

    # agent state
    cur_time: Optional[pd.Timestamp] = None
    cur_step: int = 0
    cur_tick: int = 0  # tick is the most fine-grained time unit (typically minute)
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
        assert len(self.market_price) == len(self.market_vol)
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
        self.pa_twap = price_advantage(self.exec_avg_price, self.baseline_twap, self.direction)
        self.pa_vwap = price_advantage(self.exec_avg_price, self.baseline_vwap, self.direction)
        self.fulfill_rate = (self.target - self.position) / self.target
        if abs(self.fulfill_rate - 1.0) < 1e-5:
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


@dataclass
class StepState:
    # market info and execution volume for current step
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
                                       self.episode_state.direction)
        self.pa_vwap = price_advantage(self.exec_avg_price, self.episode_state.baseline_vwap,
                                       self.episode_state.direction)


class SingleOrderEnv(gym.Env):
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

    def retrieve_backtest_data(self, field: str):
        # Retrieve backtest data for RL-specific use (including reward calculation)
        return D.features(
            [self.cur_order.stock_id],
            ['$open', '$close', '$high', '$low', '$volume'],
            start_time=self.cur_order.start_time,
            end_time=self.cur_order.end_time,
            freq=self.inner_frequency
        )[field].to_numpy()

    def initialize_state(self):
        # Synchronous state for executor to EpisodicState
        self.executor.reset(start_time=self.cur_order.start_time, end_time=self.cur_order.end_time)
        state = EpisodicState(
            stock_id=self.cur_order.stock_id,
            start_time=self.cur_order.start_time,
            end_time=self.cur_order.end_time,
            direction=self.cur_order.direction,
            target=self.cur_order.amount,
            num_step=self.executor.trade_calendar.get_trade_len(),
            market_price=self.retrieve_backtest_data('$close'),
            market_vol=self.retrieve_backtest_data('$volume'),
        )
        state.cur_step = self.executor.trade_calendar.get_trade_step()
        assert state.cur_step == 0
        state.cur_time, _ = self.executor.trade_calendar.get_step_time(state.cur_step)
        return state

    def update_state(self, exec_vol):
        # Synchronous exec_vol to executor and synchronous back to EpisodicState
        calendar = self.executor.trade_calendar
        state = self.ep_state

        trade_step = calendar.get_trade_step()
        trade_start_time, trade_end_time = calendar.get_step_time(trade_step)
        order_kwargs = asdict(self.cur_order)
        order_kwargs.update(start_time=trade_start_time, end_time=trade_end_time, amount=exec_vol)
        trade_decision = Order(**order_kwargs)
        execute_result = self.executor.execute([trade_decision])
        cur_tick = state.cur_tick

        inner_exec_vol = np.array([order.deal_amount for order, _, __, ___ in execute_result])
        ticks_this_step = len(inner_exec_vol)
        state.cur_step = trade_step = calendar.get_trade_step()
        state.cur_tick += ticks_this_step
        state.position -= np.sum(inner_exec_vol)
        state.position_history[trade_step] = state.position
        state.done = self.executor.finished()
        state.exec_vol = inner_exec_vol if state.exec_vol is None else \
            np.concatenate((state.exec_vol, inner_exec_vol))

        if state.done:
            state.update_stats()
        else:
            state.cur_time, _ = calendar.get_step_time(trade_step)

        l, r = cur_tick, cur_tick + ticks_this_step
        assert 0 <= l < r
        return StepState(inner_exec_vol, state.market_vol[l:r], state.market_price[l:r], state)

    def reset(self):
        try:
            self.cur_order = next(self.dataloader)
        except StopIteration:
            self.dataloader = None
            return None

        self.execute_result = []
        self.ep_state = self.initialize_state()

        self.action_history = np.full(self.ep_state.num_step, np.nan)
        return self.observation(self.ep_state)

    def step(self, action):
        assert self.dataloader is not None
        assert not self.executor.finished()
        self.action_history[self.ep_state.cur_step] = action

        exec_vol = self.action(action, self.ep_state)
        step_state = self.update_state(exec_vol)
        if self.executor.finished():
            assert self.ep_state.done

        reward, rew_info = self.reward(self.ep_state, step_state)

        info = {
            'action_history': self.action_history,
            'category': self.ep_state.direction,
            'reward': rew_info
        }
        if self.ep_state.done:
            info['logs'] = self.ep_state.logs()
            info['index'] = {
                'ins': self.ep_state.stock_id,
                'date': self.ep_state.start_time,
            }
            # TODO: collect logs
            pprint(info)

        return self.observation(self.ep_state), reward, self.ep_state.done, info


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
    exchange = get_exchange(
        freq="day",
        limit_threshold=0.095,
        deal_price="close",
        open_cost=0.0005,
        close_cost=0.0015,
        min_cost=5
    )

    observation = Observation(time_per_step)
    action = Action()
    reward_fn = Reward()

    def dummy_env():
        executor = get_executor(
            trade_start_time,
            trade_end_time,
            executor_config,
            exchange,
            benchmark,
            1000000000,
        )
        return SingleOrderEnv(
            observation, action, reward_fn,
            iter(DataLoader(QlibOrderDataset('rl_orders'), batch_size=None, shuffle=True)), executor)

    policy = DummyPolicy()

    # This can not be replaced with SubprocVectorEnv
    # File "/xxx/qlib/qlib/data/data.py", line 462, in dataset_processor
    # p = Pool(processes=workers)
    # AssertionError: daemonic processes are not allowed to have children
    envs = DummyVectorEnv([dummy_env for _ in range(4)])
    test_collector = Collector(policy, envs)
    policy.eval()
    # TODO: create a queue for all orders and make it auto-complete when all the orders are processed
    test_collector.collect(n_episode=10)


### This is a full RL strategy ###


class QlibOrderDataset(Dataset):
    def __init__(self, order_file):
        with open(order_file, 'rb') as f:
            self.orders = pickle.load(f)

    def __len__(self):
        return len(self.orders)

    def __getitem__(self, index):
        return self.orders[index]


class DummyPolicy(BasePolicy):
    def forward(self, batch, state=None, **kwargs):
        return Batch(act=np.random.randint(0, 5, size=(len(batch), )))

    def learn(self, *args, **kwargs):
        pass


class Observation:
    def __init__(self, time_per_step):
        self.time_per_step = time_per_step

    def __call__(self, ep_state: EpisodicState) -> Any:
        obs = self.observe(ep_state)
        if not self.validate(obs):
            raise ValueError(f'Observation space does not contain obs. Space: {self.observation_space} Sample: {obs}')
        return obs

    def validate(self, obs: Any) -> bool:
        return self.observation_space.contains(obs)

    @property
    def observation_space(self):
        space = {
            'direction': spaces.Discrete(2),
            'cur_step': spaces.Box(0, MAX_STEPS, shape=(), dtype=np.int32),
            'num_step': spaces.Box(0, MAX_STEPS, shape=(), dtype=np.int32),
            'target': spaces.Box(-1e-5, np.inf, shape=()),
            'position': spaces.Box(-1e-5, np.inf, shape=()),
            'features': spaces.Box(-np.inf, np.inf, shape=(5, ))
        }
        return spaces.Dict(space)

    def observe(self, ep_state: EpisodicState) -> Any:
        return {
            'direction': _to_int32(ep_state.direction),
            'cur_step': _to_int32(min(ep_state.cur_step, ep_state.num_step - 1)),
            'num_step': _to_int32(ep_state.num_step),
            'target': _to_float32(ep_state.target),
            'position': _to_float32(ep_state.position),
            'features': D.features(
                [ep_state.stock_id],
                ['$open', '$close', '$high', '$low', '$volume'],
                start_time=ep_state.start_time,
                end_time=ep_state.end_time,
                freq=self.time_per_step
            ).loc[(ep_state.stock_id, ep_state.cur_time)].to_numpy(),
        }


class Action:
    denominator = 4

    @property
    def action_space(self):
        return spaces.Discrete(self.denominator + 1)

    def __call__(self, action: Any, ep_state: EpisodicState) -> Any:
        if not self.validate(action):
            raise ValueError(f'Action space does not contain action. Space: {self.action_space} Sample: {action}')
        act_ = self.to_volume(action, ep_state)
        return act_

    def validate(self, action: Any) -> bool:
        return self.action_space.contains(action)

    def to_volume(self, action: Any, ep_state: EpisodicState):
        exec_vol = ep_state.position / self.denominator * action
        if ep_state.cur_step + 1 >= ep_state.num_step:
            exec_vol = ep_state.position
        # TODO: might need to check whether the stock is tradable or whether it satisfies trade unit?
        return exec_vol


class Reward:
    weight = 1.0

    def __call__(self, ep_state: EpisodicState, st_state: StepState) -> Tuple[float, Dict[str, float]]:
        rew, info = 0., {}
        if ep_state.done:
            ep_rew, ep_info = self._to_tuple(self.episode_end(ep_state))
            rew += ep_rew
            info.update({f'ep/{k}': v for k, v in ep_info.items()})
        st_rew, st_info = self._to_tuple(self.step_end(ep_state, st_state))
        rew += st_rew
        info.update({f'st/{k}': v for k, v in st_info.items()})
        return rew * self.weight, info

    @staticmethod
    def _to_tuple(x):
        if isinstance(x, tuple):
            return x
        return x, {}

    def episode_end(self, ep_state: EpisodicState) -> Tuple[float, Dict[str, float]]:
        return 0.

    def step_end(self, ep_state: EpisodicState, st_state: StepState) -> Tuple[float, Dict[str, float]]:
        assert ep_state.target > 0
        baseline_price = st_state.pa_twap
        pa = baseline_price * st_state.exec_vol.sum() / ep_state.target
        penalty = -100 * ((st_state.exec_vol / ep_state.target) ** 2).sum()  # penalize too much volume at one step
        reward = pa + penalty
        return reward, {'pa': pa, 'penalty': penalty}


def _to_int32(val): return np.array(int(val), dtype=np.int32)
def _to_float32(val): return np.array(val, dtype=np.float32)

### End of RL strategy ###


if __name__ == '__main__':
    _main()

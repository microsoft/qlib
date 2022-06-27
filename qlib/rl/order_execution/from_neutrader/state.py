import abc
import math
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Callable, Literal, Optional, Tuple

import numpy as np
import pandas as pd

EPSILON = 1e-7


class FlowDirection(str, Enum):
    ACQUIRE = "acquire"
    LIQUIDATE = "liquidate"


def _round_time(time: int, granularity: int) -> int:
    return time - time % granularity


@dataclass
class BaseEpisodicState(abc.ABC):
    """
    Base class for episodic states.
    """

    # requirements
    start_time: int
    end_time: int
    time_per_step: int
    vol_limit: Optional[float]  # TODO: meaning?
    price_func: Callable[[str], np.ndarray]  # TODO: meaning?
    volume_func: Callable[[], np.ndarray]  # TODO: meaning?
    on_step_end: Optional[Callable[..., None]]  # TODO: meaning?
    on_episode_end: Optional[Callable[..., None]]  # TODO: meaning?
    asset_num: int  # TODO: meaning?

    # agent states
    num_step: int = field(init=False)  # Number of steps
    cur_time: int = field(init=False)  # Current time
    cur_step: int = field(init=False, default=0)
    exec_vol: Optional[np.ndarray] = field(init=False, default=None)  # Execution history
    last_step_duration: int = field(init=False)
    position: float = field(init=False)
    position_history: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.cur_time = self.start_time
        rounded_start_time = _round_time(self.start_time, self.time_per_step)

        # TODO: why not rounding end time?
        self.num_step = math.floor((self.end_time - rounded_start_time) / self.time_per_step)

    def logs(self) -> dict:
        # Base logging information shared across all subclasses.
        # You can call logs = super().logs() to get these default logs and use logs.update(...) to add other logging
        # information or override it completely to remove these logging fields.
        return {
            "logs": {
                "stop_time": self.cur_time - self.start_time,
                "stop_step": self.cur_step,
            },
            "history": {
                "volume": self.execution_history(),
            },
        }

    def execution_history(self) -> np.ndarray:
        return np.pad(self.exec_vol, (0, self.end_time - self.start_time - len(self.exec_vol)))

    def next_duration(self) -> int:
        left, right = self.next_interval()
        return right - left

    def next_interval(self) -> Tuple[int, int]:
        left = _round_time(self.cur_time, self.time_per_step)
        right = left + self.time_per_step
        return max(left, self.start_time) - self.start_time, min(right, self.end_time) - self.start_time

    @classmethod
    def get_init_field_names(cls):
        ret = []
        for f in fields(cls):
            if f.init:
                ret.append(f.name)
        return ret

    @abc.abstractmethod
    def step(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def done(self) -> bool:
        return False


@dataclass
class IntraDaySingleAssetDataSchema:
    """
    In the current context, raw should be a DataFrame with `datetime` as index and
    (at least) `$vwap0`, `$volume0`, `$close0` as columns.
    `processed` should be a DataFrame of 240x6, which is the same as `processed_prev`.
    """

    date: pd.Timestamp
    stock_id: str
    start_time: int
    end_time: int
    target: float
    flow_dir: FlowDirection
    raw: pd.DataFrame
    processed: pd.DataFrame
    processed_prev: pd.DataFrame

    def get_price(self, type: Literal['deal', 'close'] = 'deal'):
        if type == 'deal':
            return self.raw['$price'].values
        elif type == 'close':
            return self.raw['$close0'].values

    def get_volume(self):
        return self.raw['$volume0'].values

    def get_processed_data(self, type: Literal['today', 'yesterday'] = 'today'):
        if type == 'today':
            return self.processed.to_numpy()
        elif type == 'yesterday':
            return self.processed_prev.to_numpy()


@dataclass
class SAOEEpisodicState(BaseEpisodicState):
    """
    Global state of the whole time horizon.
    """

    # requirements
    target: float
    target_limit: float
    flow_dir: FlowDirection

    # calculated statistics
    turnover: Optional[float] = field(init=False)
    baseline_twap: Optional[float] = field(init=False)
    baseline_vwap: Optional[float] = field(init=False)
    exec_avg_price: Optional[float] = field(init=False)
    pa_twap: Optional[float] = field(init=False)
    pa_vwap: Optional[float] = field(init=False)
    pa_close: Optional[float] = field(init=False)
    fulfill_rate: Optional[float] = field(init=False)

    market_price: np.ndarray = field(init=False)  # deal price, might be different from close
    market_close: np.ndarray = field(init=False)  # close price
    market_volume: np.ndarray = field(init=False)

    # NOTE: this is a temporary design to make it compatible with old qlib integration framework. As long as callback
    # functions are passed correctly, this field should be removed from this class.
    last_interval: Tuple[int, int] = field(default=(0, 0), init=False)

    def __post_init__(self) -> None:
        assert self.target >= 0
        assert self.asset_num == 1

        super().__post_init__()

        self.market_volume = self.volume_func()[self.start_time : self.end_time]
        self.market_price = self.price_func("deal")[self.start_time : self.end_time]
        self.market_close = self.price_func("close")[self.start_time : self.end_time]
        self.position = self.target
        self.position_history = np.full((self.num_step + 1), np.nan)
        self.position_history[0] = self.position
        self.baseline_twap = np.mean(self.market_price)
        if self.market_volume.sum() == 0:
            self.baseline_vwap = self.baseline_twap
        else:
            self.baseline_vwap = np.average(self.market_price, weights=self.market_volume)

    def update_stats(self) -> None:
        market_price = self.market_price[: len(self.exec_vol)]
        self.turnover = (self.exec_vol * market_price).sum()
        # exec_vol can be zero
        if np.isclose(self.exec_vol.sum(), 0):
            self.exec_avg_price = market_price[0]
        else:
            self.exec_avg_price = np.average(market_price, weights=self.exec_vol)

        self.pa_twap = _price_advantage(self.exec_avg_price, self.baseline_twap, self.flow_dir)
        self.pa_vwap = _price_advantage(self.exec_avg_price, self.baseline_vwap, self.flow_dir)
        close_average = np.mean(self.market_close)
        self.pa_close = _price_advantage(self.exec_avg_price, close_average, self.flow_dir)

        self.fulfill_rate = (self.target - self.position) / self.target_limit
        if abs(self.fulfill_rate - 1.0) < EPSILON:
            self.fulfill_rate = 1.0
        self.fulfill_rate *= 100

    def logs(self) -> dict:
        logs = super().logs()
        logs.update(
            {
                "logs": {
                    "turnover": self.turnover,
                    "baseline_twap": self.baseline_twap,
                    "baseline_vwap": self.baseline_vwap,
                    "exec_avg_price": self.exec_avg_price,
                    "pa_twap": self.pa_twap,
                    "pa_vwap": self.pa_vwap,
                    "pa_close": self.pa_close,
                    "ffr": self.fulfill_rate,
                }
            }
        )
        return logs

    def step(self, exec_vol: np.ndarray) -> None:
        l, r = self.next_interval()
        self.last_interval = (l, r)
        assert 0 <= l < r
        self.last_step_duration = len(exec_vol)
        self.position -= exec_vol.sum()
        assert (
            self.position > -EPSILON and (exec_vol > -EPSILON).all(),
            f"Execution volume is invalid: {exec_vol} (position = {self.position})",
        )
        self.cur_step += 1
        self.position_history[self.cur_step] = self.position
        self.cur_time += self.last_step_duration
        if self.cur_step == self.num_step:  # Should reach the end of episode
            assert self.cur_time == self.end_time
        self.exec_vol = exec_vol if self.exec_vol is None else np.concatenate((self.exec_vol, exec_vol))

        if self.on_step_end is not None:
            self.on_step_end(l, r, self)
        if self.done:
            self.update_stats()
            if self.on_episode_end is not None:
                self.on_episode_end(self)

    @property
    def done(self) -> bool:
        return self.position < EPSILON or self.cur_step == self.num_step


def _price_advantage(exec_price: float, baseline_price: float, flow: FlowDirection) -> float:
    if baseline_price == 0:
        return 0.0
    if flow == FlowDirection.ACQUIRE:
        return (1 - exec_price / baseline_price) * 10000
    else:
        return (exec_price / baseline_price - 1) * 10000

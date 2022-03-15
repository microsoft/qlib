from typing import Literal, NamedTuple

import numpy as np
import pandas as pd

from qlib.backtest import Order

from .base import Simulator


class BaseEpisodicState(abc.ABC):
    """
    Base class for episodic states.
    """

    # requirements
    start_time: int
    end_time: int
    time_per_step: int
    vol_limit: Optional[float]
    price_func: Callable[[str], np.ndarray]
    volume_func: Callable[[], np.ndarray]
    on_step_end: Optional[Callable[..., None]]
    on_episode_end: Optional[Callable[..., None]]
    asset_num: int

    # agent states
    num_step: int = field(init=False)
    cur_time: int = field(init=False)
    cur_step: int = field(init=False, default=0)
    done: bool = field(init=False, default=False)
    exec_vol: Optional[np.ndarray] = field(init=False, default=None)
    last_step_duration: int = field(init=False)
    position: np.ndarray = field(init=False)
    position_history: np.ndarray = field(init=False)

    def __post_init__(self):
        self.cur_time = self.start_time
        rounded_start_time = self.start_time - self.start_time % self.time_per_step
        self.num_step = (self.end_time - rounded_start_time - 1) // self.time_per_step + 1

    def step_logs(self) -> RecursiveDict:
        return RecursiveDict()

    def logs(self) -> RecursiveDict:
        # Base logging information shared across all subclasses.
        # You can call logs = super().logs() to get these default logs and use logs.update(...) to add other logging
        # information or override it completely to remove these logging fields.
        return RecursiveDict({
            "logs": {
                "stop_time": self.cur_time - self.start_time,
                "stop_step": self.cur_step,
            },
            "history": {
                "volume": self.execution_history(),
            },
        })

    def execution_history(self) -> np.ndarray:
        return np.pad(self.exec_vol, (0, self.end_time - self.start_time - len(self.exec_vol)))

    def next_duration(self) -> int:
        left, right = self.next_interval()
        return right - left

    def next_interval(self) -> Tuple[int, int]:
        left = self.cur_time - self.cur_time % self.time_per_step
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


class SingleAssetOrderExecution(Simulator):
    def __init__(self, initial: Order) -> None:
        pass

    def step(self, action: Any) -> None:
        raise NotImplementedError()

    def get_state(self) -> StateType:
        raise NotImplementedError()

    def done(self) -> bool:
        raise NotImplementedError()

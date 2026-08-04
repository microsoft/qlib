# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any, List, Optional, cast

import numpy as np
import pandas as pd
from gym import spaces

from qlib.rl.data.base import ProcessedDataProvider
from qlib.rl.interpreter import ActionInterpreter, StateInterpreter
from qlib.rl.algorithm_trading.state import SAATState
from qlib.typehint import TypedDict

__all__ = [
    "FullHistoryATStateInterpreter",
    "CategoricalATActionInterpreter",
    "FullHistoryATObs",
]

from qlib.utils import init_instance_by_config


def canonicalize(value: int | float | np.ndarray | pd.DataFrame | dict) -> np.ndarray | dict:
    """To 32-bit numeric types. Recursively."""
    if isinstance(value, pd.DataFrame):
        return value.to_numpy()
    if isinstance(value, (float, np.floating)) or (isinstance(value, np.ndarray) and value.dtype.kind == "f"):
        return np.array(value, dtype=np.float32)
    elif isinstance(value, (int, bool, np.integer)) or (isinstance(value, np.ndarray) and value.dtype.kind == "i"):
        return np.array(value, dtype=np.int32)
    elif isinstance(value, dict):
        return {k: canonicalize(v) for k, v in value.items()}
    else:
        return value


class FullHistoryATObs(TypedDict):
    data_processed: Any
    data_processed_prev: Any
    cur_tick: Any
    cur_step: Any
    num_step: Any
    position: Any
    position_history: Any


class FullHistoryATStateInterpreter(StateInterpreter[SAATState, FullHistoryATObs]):
    """The observation of all the history, including today (until this moment), and yesterday.

    Parameters
    ----------
    max_step
        Total number of steps (an upper-bound estimation). For example, 390min / 30min-per-step = 13 steps.
    data_ticks
        Equal to the total number of records. For example, in SAAT per minute,
        the total ticks is the length of day in minutes.
    data_dim
        Number of dimensions in data.
    processed_data_provider
        Provider of the processed data.
    """

    def __init__(
        self,
        max_step: int,
        data_ticks: int,
        data_dim: int,
        processed_data_provider: dict | ProcessedDataProvider,
    ) -> None:
        super().__init__()

        self.max_step = max_step
        self.data_ticks = data_ticks
        self.data_dim = data_dim
        self.processed_data_provider: ProcessedDataProvider = init_instance_by_config(
            processed_data_provider,
            accept_types=ProcessedDataProvider,
        )

    def interpret(self, state: SAATState) -> FullHistoryATObs:
        processed = self.processed_data_provider.get_data(
            stock_id=state.task.stock_id,
            date=pd.Timestamp(state.task.start_time.date()),
            feature_dim=self.data_dim,
            time_index=state.ticks_index,
        )

        position_history = np.full(self.max_step + 1, 0.0, dtype=np.float32)  # Initialize position is 0
        position_history[1 : len(state.history_steps) + 1] = state.history_steps["position"].to_numpy()

        # The min, slice here are to make sure that indices fit into the range,
        # even after the final step of the simulator (in the done step),
        # to make network in policy happy.
        return cast(
            FullHistoryATObs,
            canonicalize(
                {
                    "data_processed": np.array(self._mask_future_info(processed.today, state.cur_time)),
                    "data_processed_prev": np.array(processed.yesterday),
                    "cur_tick": _to_int32(min(int(np.sum(state.ticks_index < state.cur_time)), self.data_ticks - 1)),
                    "cur_step": _to_int32(min(state.cur_step, self.max_step - 1)),
                    "num_step": _to_int32(self.max_step),
                    "position": _to_float32(state.position),
                    "position_history": _to_float32(position_history[: self.max_step]),
                },
            ),
        )

    @property
    def observation_space(self) -> spaces.Dict:
        space = {
            "data_processed": spaces.Box(-np.inf, np.inf, shape=(self.data_ticks, self.data_dim)),
            "data_processed_prev": spaces.Box(-np.inf, np.inf, shape=(self.data_ticks, self.data_dim)),
            "cur_tick": spaces.Box(0, self.data_ticks - 1, shape=(), dtype=np.int32),
            "cur_step": spaces.Box(0, self.max_step - 1, shape=(), dtype=np.int32),
            # TODO: support arbitrary length index
            "num_step": spaces.Box(self.max_step, self.max_step, shape=(), dtype=np.int32),
            "position": spaces.Box(-np.inf, np.inf, shape=()),
            "position_history": spaces.Box(-np.inf, np.inf, shape=(self.max_step,)),
        }
        return spaces.Dict(space)

    @staticmethod
    def _mask_future_info(arr: pd.DataFrame, current: pd.Timestamp) -> pd.DataFrame:
        arr = arr.copy(deep=True)
        arr.loc[current:] = 0.0  # mask out data after this moment (inclusive)
        return arr


class CategoricalATActionInterpreter(ActionInterpreter[SAATState, int, float]):
    """Convert a discrete policy action to a continuous action, then multiplied by ``task.cash``.

    Parameters
    ----------
    values
        It can be a list of length $L$: $[a_1, a_2, \\ldots, a_L]$.
        Then when policy givens decision $x$, $a_x$ times order amount is the output.
        It can also be an integer $n$, in which case the list of length $n+1$ is auto-generated,
        i.e., $[0, 1/n, 2/n, \\ldots, n/n]$.
    max_step
        Total number of steps (an upper-bound estimation). For example, 390min / 30min-per-step = 13 steps.
    """

    def __init__(self, values: List[int], max_step: Optional[int] = None) -> None:
        super().__init__()

        self.action_values = values
        self.max_step = max_step

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_values))

    def interpret(self, state: SAATState, action: int) -> str:
        assert 0 <= action < len(self.action_values)
        if self.action_values[action] == -1:
            return "short"
        elif self.action_values[action] == 1:
            return "long"
        else:
            return "hold"


def _to_int32(val):
    return np.array(int(val), dtype=np.int32)


def _to_float32(val):
    return np.array(val, dtype=np.float32)

from typing import Any

import numpy as np
from gym import spaces

from qlib.rl.config import STATE_INTERPRETERS, EPSILON
from qlib.rl.interpreter import StateInterpreter

from .simulator_simple import SingleAssetOrderExecutionState


@STATE_INTERPRETERS.register_module('saoe_full_history')
class FullHistoryStateInterpreter(StateInterpreter):

    def __init__(self, max_step: int, total_time: int, data_dim: int) -> None:
        self.max_step = max_step
        self.total_time = total_time
        self.data_dim = data_dim
        self._feature_loader = []

        self.env.status

    def interpret(self, state: SingleAssetOrderExecutionState) -> Any:
        return {
            'data_processed': self._mask_future_info(_to_float32(state.get_processed_data()), state.cur_time),
            'data_processed_prev': _to_float32(state.get_processed_data('yesterday')),
            'acquiring': _to_int32(state.order.direction == state.order.BUY),
            'cur_time': _to_int32(min(state.cur_time, state.end_time - 1)),
            'cur_step': _to_int32(self.env.status.cur_step),
            'num_step': _to_int32(self.max_step),
            'target': _to_float32(state.order.amount),
            'position': _to_float32(state.position),
            'position_history': _to_float32(np.nan_to_num(state.position_history)[:-1]),  # cut the final zero
        }

    @property
    def observation_space(self):
        space = {
            'data_processed': spaces.Box(-np.inf, np.inf, shape=(self.total_time, self.data_dim)),
            'data_processed_prev': spaces.Box(-np.inf, np.inf, shape=(self.total_time, self.data_dim)),
            'acquiring': spaces.Discrete(2),
            "cur_time": spaces.Box(0, self.total_time - 1, shape=(), dtype=np.int32),
            'cur_step': spaces.Box(0, self.max_step - 1, shape=(), dtype=np.int32),
            'num_step': spaces.Box(self.max_step, self.max_step, shape=(), dtype=np.int32),
            'target': spaces.Box(-EPSILON, np.inf, shape=()),
            'position': spaces.Box(-EPSILON, np.inf, shape=()),
            'position_history': spaces.Box(-EPSILON, np.inf, shape=(self.max_step,)),
        }
        return spaces.Dict(space)

    @staticmethod
    def _mask_future_info(arr, current):
        arr = arr.copy()
        arr[current:] = 0.
        return arr

    def observe(self, sample: IntraDaySingleAssetDataSchema,
                ep_state: EpisodicState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert ep_state.cur_time <= ep_state.end_time and ep_state.cur_step <= ep_state.num_step
        obs = {
            'data_processed': self._mask_future_info(_to_float32(sample.get_processed_data()), ep_state.cur_time),
            'data_processed_prev': _to_float32(sample.get_processed_data('yesterday')),
            'acquiring': _to_int32(sample.flow_dir == FlowDirection.ACQUIRE),
            'cur_time': _to_int32(min(ep_state.cur_time, ep_state.end_time - 1)),
            'cur_step': _to_int32(min(ep_state.cur_step, ep_state.num_step - 1)),
            'num_step': _to_int32(ep_state.num_step),
            'target': _to_float32(ep_state.target),
            'position': _to_float32(ep_state.position),
            'position_history': _to_float32(np.nan_to_num(ep_state.position_history)[:-1]),  # cut the final zero
        }
        if self._feature_loader:
            obs['feature'] = self._mask_future_info(
                np.stack([f.load(sample.stock_id, sample.date, np.arange(self.max_step))
                          for f in self._feature_loader], -1), ep_state.cur_step + 1)
        return obs


@OBSERVATIONS.register_module('current_step')
class CurrentStepObservation(BaseObservation):
    def __init__(self,
                 max_step: int,
                 cached_features: Optional[List[Union[OnDemandFeatureLoader,
                                                      ClassConfig[OnDemandFeatureLoader]]]] = None):
        self.max_step = max_step
        self._feature_loader = []
        if cached_features is not None:
            self._feature_loader = [f if isinstance(f, OnDemandFeatureLoader) else f.build() for f in cached_features]

    @property
    def observation_space(self):
        space = {
            'acquiring': spaces.Discrete(2),
            'cur_step': spaces.Box(0, self.max_step - 1, shape=(), dtype=np.int32),
            'num_step': spaces.Box(self.max_step, self.max_step, shape=(), dtype=np.int32),
            'target': spaces.Box(-EPSILON, np.inf, shape=()),
            'position': spaces.Box(-EPSILON, np.inf, shape=()),
        }
        if self._feature_loader:
            space['feature'] = spaces.Box(-np.inf, np.inf, shape=(len(self._feature_loader), ))
        return spaces.Dict(space)

    def observe(self, sample: IntraDaySingleAssetDataSchema, ep_state: EpisodicState) -> Any:
        assert ep_state.cur_step <= ep_state.num_step
        obs = {
            'acquiring': _to_int32(sample.flow_dir == FlowDirection.ACQUIRE),
            'cur_step': _to_int32(min(ep_state.cur_step, ep_state.num_step - 1)),
            'num_step': _to_int32(ep_state.num_step),
            'target': _to_float32(ep_state.target),
            'position': _to_float32(ep_state.position),
        }
        if self._feature_loader:
            obs['feature'] = np.array([f.load(sample.stock_id, sample.date, ep_state.cur_step)
                                       for f in self._feature_loader])
        return obs


def _to_int32(val):
    return np.array(int(val), dtype=np.int32)


def _to_float32(val):
    return np.array(val, dtype=np.float32)

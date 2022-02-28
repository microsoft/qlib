# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from typing import Any, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
from gym import spaces
from utilsd.config import ClassConfig, PythonConfig, Registry, configclass

from .base import BaseInterpreter
from .state import EpisodicState

INT32_INF = 10 ** 9
EPSILON = 1e-5


class OBSERVATIONS(metaclass=Registry, name='observation'):
    pass


@configclass
class FeatureLoaderConfig(PythonConfig):
    path: Path
    slice: Optional[Tuple[int, int, int]] = None





class OnDemandFeatureLoader:
    def __init__(self, path: Path, slice: Optional[Tuple[int, int, int]] = None):
        self.feature_path = path
        self.slice = slice

    @staticmethod
    def _load_from_array(arr, idx):
        if isinstance(arr, pd.Series):
            arr = arr.to_numpy()
        if isinstance(idx, int):
            # sometimes the cached feature is shorter than asked ones.
            # filling it with zero.
            return arr[idx] if idx < len(arr) else 0.
        else:
            if max(idx) >= len(arr):
                return np.pad(arr, (0, max(idx) + 1 - len(arr)), constant_values=0)[idx]
            return arr[idx]

    def load(self, stock_id, date, cur_step):
        feature = pd.read_pickle(self.feature_path / f'{stock_id}.pkl')
        if self.slice is not None:
            start, end, stride = self.slice
            feature = feature.iloc[:, start:end:stride]
        arr = feature.loc[stock_id, date]
        return self._load_from_array(arr, cur_step)


class BaseObservation(abc.ABC):
    @abc.abstractproperty
    def observation_space(self) -> spaces.Space:
        raise NotImplementedError

    def __call__(self, sample: IntraDaySingleAssetDataSchema, ep_state: EpisodicState) -> Any:
        """
        This method is designed to be final and should not be overridden.
        """
        obs = self.observe(sample, ep_state)
        if not self.validate(obs):
            raise ValueError(f'Observation space does not contain obs. Space: {self.observation_space} Sample: {obs}')
        return obs

    def validate(self, obs: Any) -> bool:
        return self.observation_space.contains(obs)

    @abc.abstractmethod
    def observe(self, sample: IntraDaySingleAssetDataSchema, ep_state: EpisodicState) -> Any:
        raise NotImplementedError


@OBSERVATIONS.register_module('full_history')
class FullHistoryObservation(BaseObservation):
    def __init__(self,
                 max_step: int,
                 total_time: int,
                 data_dim: int,
                 cached_features: Optional[List[Union[OnDemandFeatureLoader,
                                                      ClassConfig[OnDemandFeatureLoader]]]] = None):
        self.max_step = max_step
        self.total_time = total_time
        self.data_dim = data_dim
        self._feature_loader = []
        if cached_features is not None:
            self._feature_loader = [f if isinstance(f, OnDemandFeatureLoader) else f.build() for f in cached_features]

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
        if self._feature_loader:
            space['feature'] = spaces.Box(-np.inf, np.inf, shape=(self.max_step, len(self._feature_loader)))
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

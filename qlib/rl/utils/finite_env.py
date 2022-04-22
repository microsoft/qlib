# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This is to support finite env in vector env.
See https://github.com/thu-ml/tianshou/issues/322 for details.
"""

import copy
import gym
import numpy as np
from typing import Any, Type

from tianshou.env import BaseVectorEnv, DummyVectorEnv, ShmemVectorEnv, SubprocVectorEnv

from qlib.typehint import Literal
from .logger import RLLogger

__all__ = [
    "generate_nan_observation",
    "check_nan_observation",
    "FiniteVectorEnv",
    "FiniteDummyVectorEnv",
    "FiniteSubprocVectorEnv",
    "FiniteShmemVectorEnv",
    'FiniteEnvType',
    'finite_env_cls',
]


FiniteEnvType = Literal['dummy', 'subproc', 'shmem']


def fill_invalid(obj):
    if isinstance(obj, (int, float, bool)):
        return fill_invalid(np.array(obj))
    if hasattr(obj, "dtype"):
        if isinstance(obj, np.ndarray):
            if np.issubdtype(obj.dtype, np.floating):
                return np.full_like(obj, np.nan)
            return np.full_like(obj, np.iinfo(obj.dtype).max)
            return obj
        # dealing with corner cases that numpy number is not supported by tianshou's sharray
        return fill_invalid(np.array(obj))
    elif isinstance(obj, dict):
        return {k: fill_invalid(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fill_invalid(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple([fill_invalid(v) for v in obj])
    raise ValueError(f"Unsupported value to fill with invalid: {obj}")


def isinvalid(arr):
    if hasattr(arr, "dtype"):
        if np.issubdtype(arr.dtype, np.floating):
            return np.isnan(arr).all()
        return (np.iinfo(arr.dtype).max == arr).all()
    if isinstance(arr, dict):
        return all([isinvalid(o) for o in arr.values()])
    if isinstance(arr, (list, tuple)):
        return all([isinvalid(o) for o in arr])
    if isinstance(arr, (int, float, bool, np.number)):
        return isinvalid(np.array(arr))
    return True


def generate_nan_observation(obs_space: gym.Space) -> Any:
    # We assume that obs is complex and there must be something like float
    # otherwise this logic will not be accurate
    sample = obs_space.sample()
    sample = fill_invalid(sample)
    return sample


def check_nan_observation(obs: Any) -> bool:
    return isinvalid(obs)


class FiniteVectorEnv(BaseVectorEnv):
    def __init__(self, logger: RLLogger, env_fns, **kwargs):
        super().__init__(env_fns, **kwargs)
        self._logger = logger
        self._alive_env_ids = set()
        self._reset_alive_envs()
        self._default_obs = self._default_info = self._default_rew = None
        self._zombie = False

    def _reset_alive_envs(self):
        if not self._alive_env_ids:
            # starting or running out
            self._alive_env_ids = set(range(self.env_num))

    # to workaround with tianshou's buffer and batch
    def _set_default_obs(self, obs):
        if obs is not None and self._default_obs is None:
            self._default_obs = copy.deepcopy(obs)

    def _set_default_info(self, info):
        if info is not None and self._default_info is None:
            self._default_info = copy.deepcopy(info)

    def _set_default_rew(self, rew):
        if rew is not None and self._default_rew is None:
            self._default_rew = copy.deepcopy(rew)

    def _get_default_obs(self):
        return copy.deepcopy(self._default_obs)

    def _get_default_info(self):
        return copy.deepcopy(self._default_info)

    def _get_default_rew(self):
        return copy.deepcopy(self._default_rew)

    # END

    def _postproc_env_obs(self, obs):
        # reserved for shmem vector env to restore empty observation
        if obs is None or check_nan_observation(obs):
            return None
        return obs

    def reset(self, id=None):
        assert not self._zombie
        id = self._wrap_id(id)
        self._reset_alive_envs()

        # ask super to reset alive envs and remap to current index
        request_id = list(filter(lambda i: i in self._alive_env_ids, id))
        obs = [None] * len(id)
        id2idx = {i: k for k, i in enumerate(id)}
        if request_id:
            for i, o in zip(request_id, super().reset(request_id)):
                obs[id2idx[i]] = self._postproc_env_obs(o)

        for i, o in zip(id, obs):
            if o is None and i in self._alive_env_ids:
                self._alive_env_ids.remove(i)

        # logging
        for i, o in zip(id, obs):
            if i in self._alive_env_ids:
                self._logger.log_reset(i, obs)

        # fill empty observation with default(fake) observation
        for o in obs:
            self._set_default_obs(o)
        for i in range(len(obs)):
            if obs[i] is None:
                obs[i] = self._get_default_obs()

        if not self._alive_env_ids:
            # comment this line so that the env becomes indisposable
            # self.reset()
            self._zombie = True
            raise StopIteration

        return np.stack(obs)

    def step(self, action, id=None):
        assert not self._zombie
        id = self._wrap_id(id)
        id2idx = {i: k for k, i in enumerate(id)}
        request_id = list(filter(lambda i: i in self._alive_env_ids, id))
        result = [[None, None, False, None] for _ in range(len(id))]

        # ask super to step alive envs and remap to current index
        if request_id:
            valid_act = np.stack([action[id2idx[i]] for i in request_id])
            for i, r in zip(request_id, zip(*super().step(valid_act, request_id))):
                result[id2idx[i]] = list(r)
                result[id2idx[i]][0] = self._postproc_env_obs(result[id2idx[i]][0])

        # logging
        for i, r in zip(id, result):
            if i in self._alive_env_ids:
                self._logger.log_step(i, *r)

        # fill empty observation/info with default(fake)
        for _, r, ___, i in result:
            self._set_default_info(i)
            self._set_default_rew(r)
        for i in range(len(result)):
            if result[i][0] is None:
                result[i][0] = self._get_default_obs()
            if result[i][1] is None:
                result[i][1] = self._get_default_rew()
            if result[i][3] is None:
                result[i][3] = self._get_default_info()

        return list(map(np.stack, zip(*result)))


class FiniteDummyVectorEnv(FiniteVectorEnv, DummyVectorEnv):
    pass


class FiniteSubprocVectorEnv(FiniteVectorEnv, SubprocVectorEnv):
    pass


class FiniteShmemVectorEnv(FiniteVectorEnv, ShmemVectorEnv):
    pass


def finite_env_cls(env_type: FiniteEnvType) -> Type[FiniteVectorEnv]:
    if env_type == 'dummy':
        return FiniteDummyVectorEnv
    elif env_type == 'subproc':
        return FiniteSubprocVectorEnv
    elif env_type == 'shmem':
        return FiniteShmemVectorEnv
    raise ValueError(f'Unexpected env_type: {env_type}')

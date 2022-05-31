# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import Counter

import gym
import numpy as np
from tianshou.data import Batch, Collector
from tianshou.policy import BasePolicy
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from qlib.rl.utils.finite_env import (
    LogWriter,
    FiniteDummyVectorEnv,
    FiniteShmemVectorEnv,
    FiniteSubprocVectorEnv,
    check_nan_observation,
    generate_nan_observation,
)


_test_space = gym.spaces.Dict(
    {
        "sensors": gym.spaces.Dict(
            {
                "position": gym.spaces.Box(low=-100, high=100, shape=(3,)),
                "velocity": gym.spaces.Box(low=-1, high=1, shape=(3,)),
                "front_cam": gym.spaces.Tuple(
                    (gym.spaces.Box(low=0, high=1, shape=(10, 10, 3)), gym.spaces.Box(low=0, high=1, shape=(10, 10, 3)))
                ),
                "rear_cam": gym.spaces.Box(low=0, high=1, shape=(10, 10, 3)),
            }
        ),
        "ext_controller": gym.spaces.MultiDiscrete((5, 2, 2)),
        "inner_state": gym.spaces.Dict(
            {
                "charge": gym.spaces.Discrete(100),
                "system_checks": gym.spaces.MultiBinary(10),
                "job_status": gym.spaces.Dict(
                    {
                        "task": gym.spaces.Discrete(5),
                        "progress": gym.spaces.Box(low=0, high=100, shape=()),
                    }
                ),
            }
        ),
    }
)


class FiniteEnv(gym.Env):
    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.loader = DataLoader(dataset, sampler=DistributedSampler(dataset, num_replicas, rank), batch_size=None)
        self.iterator = None
        self.observation_space = gym.spaces.Discrete(255)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            self.current_sample, self.step_count = next(self.iterator)
            self.current_step = 0
            return self.current_sample
        except StopIteration:
            self.iterator = None
            return generate_nan_observation(self.observation_space)

    def step(self, action):
        self.current_step += 1
        assert self.current_step <= self.step_count
        return (
            0,
            1.0,
            self.current_step >= self.step_count,
            {"sample": self.current_sample, "action": action, "metric": 2.0},
        )


class FiniteEnvWithComplexObs(FiniteEnv):
    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.loader = DataLoader(dataset, sampler=DistributedSampler(dataset, num_replicas, rank), batch_size=None)
        self.iterator = None
        self.observation_space = gym.spaces.Discrete(255)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            self.current_sample, self.step_count = next(self.iterator)
            self.current_step = 0
            return _test_space.sample()
        except StopIteration:
            self.iterator = None
            return generate_nan_observation(self.observation_space)

    def step(self, action):
        self.current_step += 1
        assert self.current_step <= self.step_count
        return (
            _test_space.sample(),
            1.0,
            self.current_step >= self.step_count,
            {"sample": _test_space.sample(), "action": action, "metric": 2.0},
        )


class DummyDataset(Dataset):
    def __init__(self, length):
        self.length = length
        self.episodes = [3 * i % 5 + 1 for i in range(self.length)]

    def __getitem__(self, index):
        assert 0 <= index < self.length
        return index, self.episodes[index]

    def __len__(self):
        return self.length


class AnyPolicy(BasePolicy):
    def forward(self, batch, state=None):
        return Batch(act=np.stack([1] * len(batch)))

    def learn(self, batch):
        pass


def _finite_env_factory(dataset, num_replicas, rank, complex=False):
    if complex:
        return lambda: FiniteEnvWithComplexObs(dataset, num_replicas, rank)
    return lambda: FiniteEnv(dataset, num_replicas, rank)


class MetricTracker(LogWriter):
    def __init__(self, length):
        super().__init__()
        self.counter = Counter()
        self.finished = set()
        self.length = length

    def on_env_step(self, env_id, obs, rew, done, info):
        assert rew == 1.0
        index = info["sample"]
        if done:
            # assert index not in self.finished
            self.finished.add(index)
        self.counter[index] += 1

    def validate(self):
        assert len(self.finished) == self.length
        for k, v in self.counter.items():
            assert v == k * 3 % 5 + 1


class DoNothingTracker(LogWriter):
    def on_env_step(self, *args, **kwargs):
        pass


def test_finite_dummy_vector_env():
    length = 100
    dataset = DummyDataset(length)
    envs = FiniteDummyVectorEnv(MetricTracker(length), [_finite_env_factory(dataset, 5, i) for i in range(5)])
    envs._collector_guarded = True
    policy = AnyPolicy()
    test_collector = Collector(policy, envs, exploration_noise=True)

    for _ in range(1):
        envs._logger = [MetricTracker(length)]
        try:
            test_collector.collect(n_step=10**18)
        except StopIteration:
            envs._logger[0].validate()


def test_finite_shmem_vector_env():
    length = 100
    dataset = DummyDataset(length)
    envs = FiniteShmemVectorEnv(MetricTracker(length), [_finite_env_factory(dataset, 5, i) for i in range(5)])
    envs._collector_guarded = True
    policy = AnyPolicy()
    test_collector = Collector(policy, envs, exploration_noise=True)

    for _ in range(1):
        envs._logger = [MetricTracker(length)]
        try:
            test_collector.collect(n_step=10**18)
        except StopIteration:
            envs._logger[0].validate()


def test_finite_subproc_vector_env():
    length = 100
    dataset = DummyDataset(length)
    envs = FiniteSubprocVectorEnv(MetricTracker(length), [_finite_env_factory(dataset, 5, i) for i in range(5)])
    envs._collector_guarded = True
    policy = AnyPolicy()
    test_collector = Collector(policy, envs, exploration_noise=True)

    for _ in range(1):
        envs._logger = [MetricTracker(length)]
        try:
            test_collector.collect(n_step=10**18)
        except StopIteration:
            envs._logger[0].validate()


def test_nan():
    assert check_nan_observation(generate_nan_observation(_test_space))
    assert not check_nan_observation(_test_space.sample())


def test_finite_dummy_vector_env_complex():
    length = 100
    dataset = DummyDataset(length)
    envs = FiniteDummyVectorEnv(
        DoNothingTracker(), [_finite_env_factory(dataset, 5, i, complex=True) for i in range(5)]
    )
    envs._collector_guarded = True
    policy = AnyPolicy()
    test_collector = Collector(policy, envs, exploration_noise=True)

    try:
        test_collector.collect(n_step=10**18)
    except StopIteration:
        pass


def test_finite_shmem_vector_env_complex():
    length = 100
    dataset = DummyDataset(length)
    envs = FiniteShmemVectorEnv(
        DoNothingTracker(), [_finite_env_factory(dataset, 5, i, complex=True) for i in range(5)]
    )
    envs._collector_guarded = True
    policy = AnyPolicy()
    test_collector = Collector(policy, envs, exploration_noise=True)

    try:
        test_collector.collect(n_step=10**18)
    except StopIteration:
        pass

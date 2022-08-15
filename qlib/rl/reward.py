# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Generic, Optional, Tuple, TypeVar

from qlib.typehint import final

if TYPE_CHECKING:
    from .utils.env_wrapper import EnvWrapper

SimulatorState = TypeVar("SimulatorState")


class Reward(Generic[SimulatorState]):
    """
    Reward calculation component that takes a single argument: state of simulator. Returns a real number: reward.

    Subclass should implement ``reward(simulator_state)`` to implement their own reward calculation recipe.
    """

    env: Optional[EnvWrapper] = None

    @final
    def __call__(self, simulator_state: SimulatorState) -> float:
        return self.reward(simulator_state)

    def reward(self, simulator_state: SimulatorState) -> float:
        """Implement this method for your own reward."""
        raise NotImplementedError("Implement reward calculation recipe in `reward()`.")

    def log(self, name: str, value: Any) -> None:
        assert self.env is not None
        self.env.logger.add_scalar(name, value)


class RewardCombination(Reward):
    """Combination of multiple reward."""

    def __init__(self, rewards: Dict[str, Tuple[Reward, float]]) -> None:
        self.rewards = rewards

    def reward(self, simulator_state: Any) -> float:
        total_reward = 0.0
        for name, (reward_fn, weight) in self.rewards.items():
            rew = reward_fn(simulator_state) * weight
            total_reward += rew
            self.log(name, rew)
        return total_reward


# TODO:
# reward_factory is disabled for now

# _RegistryConfigReward = RegistryConfig[REWARDS]


# @configclass
# class _WeightedRewardConfig:
#     weight: float
#     reward: _RegistryConfigReward


# RewardConfig = Union[_RegistryConfigReward, Dict[str, Union[_RegistryConfigReward, _WeightedRewardConfig]]]


# def reward_factory(reward_config: RewardConfig) -> Reward:
#     """
#     Use this factory to instantiate the reward from config.
#     Simply using ``reward_config.build()`` might not work because reward can have complex combinations.
#     """
#     if isinstance(reward_config, dict):
#         # as reward combination
#         rewards = {}
#         for name, rew in reward_config.items():
#             if not isinstance(rew, _WeightedRewardConfig):
#                 # default weight is 1.
#                 rew = _WeightedRewardConfig(weight=1., rew=rew)
#             # no recursive build in this step
#             rewards[name] = (rew.reward.build(), rew.weight)
#         return RewardCombination(rewards)
#     else:
#         # single reward
#         return reward_config.build()

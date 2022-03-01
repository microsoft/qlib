import weakref

from enum import Enum
from typing import Dict, Tuple, Union, Optional, Any

from qlib.rl.config import REWARDS, RegistryConfig, configclass


class BaseReward:
    """
    Reward calculation component that takes a single argument: state of simulator. Returns a number: reward.

    Subclass should implement ``reward(simulator_state)`` to implement their own reward calculation recipe.
    """

    env_wrapper: Optional[weakref.ReferenceType['qlib.rl.utils.env_wrapper.EnvWrapper']] = None

    def __call__(self, simulator_state: Any) -> float:
        return self.reward(simulator_state)

    def reward(self, simulator_state: Any) -> float:
        raise NotImplementedError('Implement reward calculation recipe in `reward()`.')

    def log(self, name, value):
        self.env_wrapper.logger.add_scalar(name, value)


class RewardCombination(BaseReward):
    def __init__(self, rewards: Dict[str, Tuple[BaseReward, float]]):
        self.rewards: Dict[str, BaseReward] = rewards

    def reward(self, simulator_state: Any) -> float:
        total_reward = 0.
        for name, (reward_fn, weight) in self.rewards.items():
            rew = reward_fn(simulator_state) * weight
            total_reward += rew
            self.log(name, rew)
        return total_reward


_RegistryConfigReward = RegistryConfig[REWARDS]


@configclass
class _WeightedRewardConfig:
    weight: float
    reward: _RegistryConfigReward


RewardConfig = Union[_RegistryConfigReward, Dict[str, Union[_RegistryConfigReward, _WeightedRewardConfig]]]


def reward_factory(reward_config: RewardConfig) -> BaseReward:
    """
    Use this factory to instantiate the reward from config.
    Simply using ``reward_config.build()`` might not work because reward can have complex combinations.
    """
    if isinstance(reward_config, dict):
        # as reward combination
        rewards = {}
        for name, rew in reward_config.items():
            if not isinstance(rew, _WeightedRewardConfig):
                # default weight is 1.
                rew = _WeightedRewardConfig(weight=1., rew=rew)
            # no recursive build in this step
            rewards[name] = (rew.reward.build(), rew.weight)
        return RewardCombination(rewards)
    else:
        # single reward
        return reward_config.build()

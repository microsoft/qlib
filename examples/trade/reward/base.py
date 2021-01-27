import numpy as np


class Abs_Reward(object):
    """The abstract class for Reward."""

    def __init__(self, config):
        return

    def get_reward(self):
        """:return: reward"""
        reward = 0
        return reward

    def __call__(self, *args, **kargs):
        return self.get_reward(*args, **kargs)

    def isinstant(self):
        """:return: Whether the reward should be given at every timestep or only at the end of this episode."""
        raise NotImplementedError


class Instant_Reward(Abs_Reward):
    def __init__(self, config):
        self.ffr_ratio = config["ffr_ratio"]
        self.vvr_ratio = config["vvr_ratio"]

    def isinstant(self):
        return True


class EndEpisode_Reward(Abs_Reward):
    def __init__(self, config):
        self.ffr_ratio = config["ffr_ratio"]
        self.vvr_ratio = config["vvr_ratio"]

    def isinstant(self):
        return False

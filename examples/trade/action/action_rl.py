import numpy as np
from gym.spaces import Discrete, Box, Tuple, MultiDiscrete

from .base import Base_Action


class Static_Action(Base_Action):
    """ """

    def __init__(self, config):
        self.action_num = config["action_num"]
        self.action_map = config["action_map"]

    def get_space(self):
        """ """
        return Discrete(self.action_num)

    def get_action(self, action, target, position, **kargs):
        """

        :param action:
        :param position:
        :param target:
        :param **kargs:

        """
        return min(target * self.action_map[action], position)

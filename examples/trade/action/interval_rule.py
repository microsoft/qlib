import numpy as np
from gym.spaces import Discrete, Box, Tuple, MultiDiscrete

from .base import Base_Action


class Rule_Static_Interval(Base_Action):
    """ """

    def get_space(self):
        """ """
        return Box(0, np.inf, shape=(), dtype=np.float32)

    def get_action(self, action, target, position, interval_num, interval, **kargs):
        """

        :param action: param target:
        :param position: param interval_num:
        :param interval: param **kargs:
        :param target:
        :param interval_num:
        :param **kargs:

        """
        return target / (interval_num) * action


class Rule_Dynamic_Interval(Base_Action):
    """ """

    def get_space(self):
        """ """
        return Box(0, np.inf, shape=(), dtype=np.float32)

    def get_action(self, action, target, position, interval_num, interval, **kargs):
        """

        :param action: param target:
        :param position: param interval_num:
        :param interval: param **kargs:
        :param target:
        :param interval_num:
        :param **kargs:

        """
        return position / (interval_num - interval) * action

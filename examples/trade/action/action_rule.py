import numpy as np
from gym.spaces import Discrete, Box, Tuple, MultiDiscrete

from .base import Base_Action


class Rule_Dynamic(Base_Action):
    """ """

    def get_space(self):
        """ """
        return Box(0, np.inf, shape=(), dtype=np.float32)

    def get_action(self, action, target, position, max_step_num, t, **kargs):
        """

        :param action: param target:
        :param position: param max_step_num:
        :param t: param **kargs:
        :param target:
        :param max_step_num:
        :param **kargs:

        """
        return position / (max_step_num - (t + 1)) * action


class Rule_Static(Base_Action):
    """ """

    def get_space(self):
        """ """
        return Box(0, np.inf, shape=(), dtype=np.float32)

    def get_action(self, action, target, position, max_step_num, t, **kargs):
        """

        :param action: param target:
        :param position: param max_step_num:
        :param t: param **kargs:
        :param target:
        :param max_step_num:
        :param **kargs:

        """
        return target / max_step_num * action

import numpy as np
from gym.spaces import Discrete, Box, Tuple, MultiDiscrete


class Base_Action(object):
    """ """

    def __init__(self, config):
        return

    def __call__(self, *args, **kargs):
        return self.get_action(*args, **kargs)

    def get_action(self, action):
        """

        :param action:

        """
        return action

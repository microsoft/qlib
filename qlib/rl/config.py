from utilsd.config import RegistryConfig, Registry, configclass

__all__ = ['RegistryConfig', 'configclass', 'REWARDS', 'OBSERVATIONS', 'ACTIONS']


EPSILON = 1E-6
"""A globally shared constant to ensure safer comparisons between floating numbers."""

# Registry: giving each class an alias to make python instance easier to config with declarative languages


class REWARDS(metaclass=Registry, name='reward'):
    pass


class STATE_INTERPRETERS(metaclass=Registry, name='state_interpreter'):
    pass


class ACTION_INTERPRETERS(metaclass=Registry, name='action_interpreter'):
    pass


class NETWORKS(metaclass=Registry, name='networks'):
    pass

from utilsd.config import RegistryConfig, Registry, configclass

__all__ = ['RegistryConfig', 'configclass', 'REWARDS', 'OBSERVATIONS', 'ACTIONS']

# Registry: giving each class an alias to make python instance easier to config with declarative languages


class REWARDS(metaclass=Registry, name='reward'):
    pass


class OBSERVATIONS(metaclass=Registry, name='observation'):
    pass


class ACTIONS(metaclass=Registry, name='action'):
    pass

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Generic, TYPE_CHECKING, TypeVar, final
from weakref import ReferenceType

from .simulator import StateType

if TYPE_CHECKING:
    from .utils.env_wrapper import EnvWrapper


__all__ = ["AuxiliaryInfoCollector"]

AuxInfoType = TypeVar("AuxInfoType")


class AuxiliaryInfoCollector(Generic[StateType, AuxInfoType]):
    """Override this class to collect customized auxiliary information from environment."""

    _env: ReferenceType["EnvWrapper"]

    @property
    def env(self) -> "EnvWrapper":
        e = self._env()
        if e is None:
            raise TypeError("env can not be None")
        return e

    @final
    def __call__(self, simulator_state: StateType) -> AuxInfoType:
        return self.collect(simulator_state)

    def collect(self, simulator_state: StateType) -> AuxInfoType:
        """Override this for customized auxiliary info.
        Usually useful in Multi-agent RL.

        Parameters
        ----------
        simulator_state
            Retrieved with ``simulator.get_state()``.

        Returns
        -------
        Auxiliary information.
        """
        raise NotImplementedError("collect is not implemented!")

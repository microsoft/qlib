# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Optional, TypeVar

from qlib.typehint import final

from .simulator import StateType

if TYPE_CHECKING:
    from .utils.env_wrapper import EnvWrapper


__all__ = ["AuxiliaryInfoCollector"]

AuxInfoType = TypeVar("AuxInfoType")


class AuxiliaryInfoCollector(Generic[StateType, AuxInfoType]):
    """Override this class to collect customized auxiliary information from environment."""

    env: Optional[EnvWrapper] = None

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

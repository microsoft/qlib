# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any, Generic, TypeVar

import gymnasium as gymnasium
import numpy as np
from gymnasium import spaces

from qlib.typehint import final
from .simulator import ActType, StateType

ObsType = TypeVar("ObsType")
PolicyActType = TypeVar("PolicyActType")


class Interpreter:
    """Interpreter is a media between states produced by simulators and states needed by RL policies.
    Interpreters are two-way:

    1. From simulator state to policy state (aka observation), see :class:`StateInterpreter`.
    2. From policy action to action accepted by simulator, see :class:`ActionInterpreter`.

    Inherit one of the two sub-classes to define your own interpreter.
    This super-class is only used for isinstance check.

    Interpreters are recommended to be stateless, meaning that storing temporary information with ``self.xxx``
    in interpreter is anti-pattern. In future, we might support register some interpreter-related
    states by calling ``self.env.register_state()``, but it's not planned for first iteration.
    """


class StateInterpreter(Generic[StateType, ObsType], Interpreter):
    """State Interpreter that interpret execution result of qlib executor into rl env state"""

    @property
    def observation_space(self) -> gymnasium.Space:
        raise NotImplementedError()

    @final  # no overridden
    def __call__(self, simulator_state: StateType) -> ObsType:
        obs = self.interpret(simulator_state)
        self.validate(obs)
        return obs

    def validate(self, obs: ObsType) -> None:
        """Validate whether an observation belongs to the pre-defined observation space."""
        _gymnasium_space_contains(self.observation_space, obs)

    def interpret(self, simulator_state: StateType) -> ObsType:
        """Interpret the state of simulator.

        Parameters
        ----------
        simulator_state
            Retrieved with ``simulator.get_state()``.

        Returns
        -------
        State needed by policy. Should conform with the state space defined in ``observation_space``.
        """
        raise NotImplementedError("interpret is not implemented!")


class ActionInterpreter(Generic[StateType, PolicyActType, ActType], Interpreter):
    """Action Interpreter that interpret rl agent action into qlib orders"""

    @property
    def action_space(self) -> gymnasium.Space:
        raise NotImplementedError()

    @final  # no overridden
    def __call__(self, simulator_state: StateType, action: PolicyActType) -> ActType:
        self.validate(action)
        obs = self.interpret(simulator_state, action)
        return obs

    def validate(self, action: PolicyActType) -> None:
        """Validate whether an action belongs to the pre-defined action space."""
        _gymnasium_space_contains(self.action_space, action)

    def interpret(self, simulator_state: StateType, action: PolicyActType) -> ActType:
        """Convert the policy action to simulator action.

        Parameters
        ----------
        simulator_state
            Retrieved with ``simulator.get_state()``.
        action
            Raw action given by policy.

        Returns
        -------
        The action needed by simulator,
        """
        raise NotImplementedError("interpret is not implemented!")


def _gymnasium_space_contains(space: gymnasium.Space, x: Any) -> None:
    """Strengthened version of gymnasium.Space.contains.
    Giving more diagnostic information on why validation fails.

    Throw exception rather than returning true or false.
    """
    if isinstance(space, spaces.Dict):
        if not isinstance(x, dict) or len(x) != len(space):
<<<<<<< HEAD
            raise gymnasiumSpaceValidationError("Sample must be a dict with same length as space.", space, x)
        for k, subspace in space.spaces.items():
            if k not in x:
                raise gymnasiumSpaceValidationError(f"Key {k} not found in sample.", space, x)
            try:
                _gymnasium_space_contains(subspace, x[k])
            except gymnasiumSpaceValidationError as e:
                raise gymnasiumSpaceValidationError(f"Subspace of key {k} validation error.", space, x) from e
=======
            raise gymnasiumSpaceValidationError(
                "Sample must be a dict with same length as space.", space, x
            )
        for k, subspace in space.spaces.items():
            if k not in x:
                raise gymnasiumSpaceValidationError(
                    f"Key {k} not found in sample.", space, x
                )
            try:
                _gymnasium_space_contains(subspace, x[k])
            except gymnasiumSpaceValidationError as e:
                raise gymnasiumSpaceValidationError(
                    f"Subspace of key {k} validation error.", space, x
                ) from e
>>>>>>> f180e36a (fix: migrate from gym to gymnasium for NumPy 2.0+ compatibility)

    elif isinstance(space, spaces.Tuple):
        if isinstance(x, (list, np.ndarray)):
            x = tuple(x)  # Promote list and ndarray to tuple for contains check
        if not isinstance(x, tuple) or len(x) != len(space):
<<<<<<< HEAD
            raise gymnasiumSpaceValidationError("Sample must be a tuple with same length as space.", space, x)
=======
            raise gymnasiumSpaceValidationError(
                "Sample must be a tuple with same length as space.", space, x
            )
>>>>>>> f180e36a (fix: migrate from gym to gymnasium for NumPy 2.0+ compatibility)
        for i, (subspace, part) in enumerate(zip(space, x)):
            try:
                _gymnasium_space_contains(subspace, part)
            except gymnasiumSpaceValidationError as e:
<<<<<<< HEAD
                raise gymnasiumSpaceValidationError(f"Subspace of index {i} validation error.", space, x) from e

    else:
        if not space.contains(x):
            raise gymnasiumSpaceValidationError("Validation error reported by gymnasium.", space, x)
=======
                raise gymnasiumSpaceValidationError(
                    f"Subspace of index {i} validation error.", space, x
                ) from e

    else:
        if not space.contains(x):
            raise gymnasiumSpaceValidationError(
                "Validation error reported by gymnasium.", space, x
            )
>>>>>>> f180e36a (fix: migrate from gym to gymnasium for NumPy 2.0+ compatibility)


class gymnasiumSpaceValidationError(Exception):
    def __init__(self, message: str, space: gymnasium.Space, x: Any) -> None:
        self.message = message
        self.space = space
        self.x = x

    def __str__(self) -> str:
        return f"{self.message}\n  Space: {self.space}\n  Sample: {self.x}"

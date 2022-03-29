# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annoations

from typing import Any, final, TYPE_CHECKING
from weakref import ReferenceType

if TYPE_CHECKING:
    from .utils.env_wrapper import EnvWrapper

import gym

class Interpreter:
    """Interpreter is a media between states produced by simulators and states needed by RL policies.
    Interpreters are two-way:

    1. From simulator state to policy state (aka observation), see :class:`StateInterpreter`.
    2. From policy action to action accepted by simulator, see :class:`ActionInterpreter`.

    Inherit one of the two sub-classes to define your own interpreter.
    """

    def interpret(self, **kwargs):
        """Perfrom interpret action. Arguments differ in different contexts."""
        raise NotImplementedError("interpret is not implemented!")

    def __call__(self, **kwargs):
        """Use ``intepreter(to_be_interpret)`` to interpret a message.
        It calls the ``interpret`` method.
        """
        self.interpret(**kwargs)


class StateInterpreter(Interpreter):
    """State Interpreter that interpret execution result of qlib executor into rl env state"""

    env: ReferenceType['EnvWrapper']

    @property
    def observation_space(self) -> gym.Space:
        raise NotImplementedError()

    @final  # no overridden
    def __call__(self, simulator_state: Any) -> Any:
        obs = self.interpret(simulator_state)
        if not self.validate(obs):
            raise ValueError(f'Observation space does not contain obs.\n  Space: {self.observation_space}\n  Sample: {obs}')
        return obs

    def validate(self, obs: Any) -> bool:
        """Validate whether an observation belongs to the pre-defined observation space.""" 
        return self.observation_space.contains(obs)

    def interpret(self, simulator_state: Any) -> Any:
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


class ActionInterpreter(Interpreter):
    """Action Interpreter that interpret rl agent action into qlib orders"""

    env: ReferenceType['EnvWrapper']

    @property
    def action_space(self) -> gym.Space:
        raise NotImplementedError()

    @final  # no overridden
    def __call__(self, simulator_state: Any, action: Any) -> Any:
        if not self.validate(action):
            raise ValueError(f'Action space does not contain action.\n  Space: {self.action_space}\n  Sample: {action}')
        obs = self.interpret(simulator_state, action)
        return obs

    def validate(self, action: Any) -> bool:
        """Validate whether an action belongs to the pre-defined action space.""" 
        return self.action_space.contains(action)

    def interpret(self, simulator_state: Any, action: Any) -> Any:
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

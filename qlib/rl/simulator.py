# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Optional, TypeVar

from .seed import InitialStateType

if TYPE_CHECKING:
    from .utils.env_wrapper import EnvWrapper

StateType = TypeVar("StateType")
"""StateType stores all the useful data in the simulation process
(as well as utilities to generate/retrieve data when needed)."""

ActType = TypeVar("ActType")
"""This ActType is the type of action at the simulator end."""


class Simulator(Generic[InitialStateType, StateType, ActType]):
    """
    Simulator that resets with ``__init__``, and transits with ``step(action)``.

    To make the data-flow clear, we make the following restrictions to Simulator:

    1. The only way to modify the inner status of a simulator is by using ``step(action)``.
    2. External modules can *read* the status of a simulator by using ``simulator.get_state()``,
       and check whether the simulator is in the ending state by calling ``simulator.done()``.

    A simulator is defined to be bounded with three types:

    - *InitialStateType* that is the type of the data used to create the simulator.
    - *StateType* that is the type of the **status** (state) of the simulator.
    - *ActType* that is the type of the **action**, which is the input received in each step.

    Different simulators might share the same StateType. For example, when they are dealing with the same task,
    but with different simulation implementation. With the same type, they can safely share other components in the MDP.

    Simulators are ephemeral. The lifecycle of a simulator starts with an initial state, and ends with the trajectory.
    In another word, when the trajectory ends, simulator is recycled.
    If simulators want to share context between (e.g., for speed-up purposes),
    this could be done by accessing the weak reference of environment wrapper.

    Attributes
    ----------
    env
        A reference of env-wrapper, which could be useful in some corner cases.
        Simulators are discouraged to use this, because it's prone to induce errors.
    """

    env: Optional[EnvWrapper] = None

    def __init__(self, initial: InitialStateType, **kwargs: Any) -> None:
        pass

    def step(self, action: ActType) -> None:
        """Receives an action of ActType.

        Simulator should update its internal state, and return None.
        The updated state can be retrieved with ``simulator.get_state()``.
        """
        raise NotImplementedError()

    def get_state(self) -> StateType:
        raise NotImplementedError()

    def done(self) -> bool:
        """Check whether the simulator is in a "done" state.
        When simulator is in a "done" state,
        it should no longer receives any ``step`` request.
        As simulators are ephemeral, to reset the simulator,
        the old one should be destroyed and a new simulator can be created.
        """
        raise NotImplementedError()

import weakref
from typing import Any, Optional, TypeVar, Generic

import gym

InitialStateType = TypeVar('InitialStateType')
StateType = TypeVar('StateType')

SimulatorType = TypeVar('SimulatorType')

class Simulator(Generic[InitialStateType, StateType]):
    """
    Simulator that resets with ``__init__``, and transits with ``step(action)``.
    FIXME: TBD



    To make the data-flow clear, we make the following restrictions to Simulator:

    1. The only way to modify the inner status of a simulator is by using ``step(action)``.
    2. External modules can *read* the status of a simulator by using ``simulator.get_state()`` or ``simulator.done()``.

    A simulator is defined to be bounded with two types:

    - *InitialStateType* that is the type of the data used to create the simulator.
    - *StateType* that is the type of the **status** (state) of the simulator.

    Different simulators might share the same StateType. For example, when they are dealing with the same task,
    but with different simulation implementation. With the same type, they can safely share other components in the MDP.

    Simulators are ephemeral. The lifecycle of a simulator starts with an initial state, and ends with the trajectory.
    In another word, when the trajectory ends, simulator is recycled.
    If simulators want to share context between (e.g., for speed-up purposes),
    this could be done by accessing the weak reference of environment wrapper.

    Attributes
    ----------
    history : list of Simulator


    """

    history_states: bool = True
    env_wrapper: Optional[weakref.ReferenceType['qlib.rl.utils.env_wrapper.EnvWrapper']] = None

    def __init__(self, initial: InitialStateType) -> None:
        pass

    def step(self, action: Any) -> None:
        raise NotImplementedError()

    @classmethod
    def load_state(cls: SimulatorType, state: StateType) -> SimulatorType:
        raise NotImplementedError()

    def get_state(self) -> StateType:
        raise NotImplementedError()

    def done(self) -> bool:
        raise NotImplementedError()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import weakref
from typing import Callable, Any, Iterator, Optional, NamedTuple, TypedDict

import numpy as np
import gym

from qlib.rl.aux_info import AuxiliaryInfoCollector, LogCollector
from qlib.rl.simulator import Simulator, InitialStateType, StateType, ActType
from qlib.rl.interpreter import StateInterpreter, ActionInterpreter, PolicyActType, ObsType
from qlib.rl.reward import Reward

from .finite_env import generate_nan_observation

# in this case, there won't be any seed for simulator
SEED_INTERATOR_MISSING = '_missing_'

class InfoDict(TypedDict):
    log: dict[str, Any]     # collected by LogCollector
    aux_info: dict          # Any information depends on auxiliary info collector


class EnvWrapperStatus(NamedTuple):
    """
    This is the status data structure used in EnvWrapper.
    The fields here are in the semantics of RL.
    For example, ``obs`` means the observation fed into policy.
    ``action`` means the raw action returned by policy.
    """
    cur_step: int
    done: bool
    initial_state: Optional[Any]
    obs_history: Optional[list[np.ndarray]]
    action_history: Optional[list[np.ndarray]]
    reward_history: Optional[list[np.ndarray]]


class EnvWrapper(gym.Env[ObsType, PolicyActType]):
    """Qlib-based RL environment.
    This is a wrapper of components, including simulator, state-interpreter, action-interpreter, reward.

    FIXME: TBD


    Attributes
    ----------
    status : EnvWrapperStatus
        Status indicator. All terms are in *RL language*.
        It can be used if users care about data on the RL side.
        Can be none when no trajectory is available.
    """

    def __init__(
        self,
        simulator_fn: Callable[[InitialStateType], Simulator],
        state_interpreter: StateInterpreter[StateType, ObsType],
        action_interpreter: ActionInterpreter[StateType, PolicyActType, ActType],
        seed_iterator: Optional[Iterator[InitialStateType]],
        reward_fn: Optional[Reward] = None,
        log_collector: Optional[LogCollector] = None,
        aux_info_collector: Optional[AuxiliaryInfoCollector] = None
    ):
        # assign weak reference to wrapper
        for obj in [state_interpreter, action_interpreter, reward_fn, aux_info_collector]:
            obj.env_wrapper = weakref.ref(self)

        self.simulator_fn = simulator_fn
        self.state_interpreter = state_interpreter
        self.action_interpreter = action_interpreter

        if seed_iterator is None:
            # in this case, there won't be any seed for simulator
            self.seed_iterator = SEED_INTERATOR_MISSING
        else:
            self.seed_iterator = seed_iterator
        self.reward_fn = reward_fn

        self.log_collector = log_collector
        self.aux_info_collector = aux_info_collector

        self.status: Optional[EnvWrapperStatus] = None

    @property
    def action_space(self):
        return self.action_interpreter.action_space

    @property
    def observation_space(self):
        return self.state_interpreter.observation_space

    def reset(self, **kwargs) -> ObsType:
        # Try to get a state from state queue, and init the simulator with this state.
        # If the queue is exhausted, generate an invalid (nan) observation
        try:
            if self.seed_iterator is None:
                raise RuntimeError('You can trying to get a state from a dead environment wrapper.')

            # FIXME: simulator/observation might need seed to prefetch something
            # as only seed has the ability to do the work beforehands

            if self.seed_iterator is SEED_INTERATOR_MISSING:
                # no initial state
                self.simulator = self.simulator_fn()
            else:
                initial_state = next(self.seed_iterator)
                self.simulator = self.simulator_fn(initial_state)
                sim_state = self.simulator.get_state()
                obs = self.state_interpreter(sim_state)

            self.status = EnvWrapperStatus(
                cur_step=0,
                done=False,
                initial_state=initial_state,
                obs_history=[obs],
                action_history=[],
                reward_history=[]
            )

            return obs

        except StopIteration:
            # The environment should be recycled because it's in a dead state.
            self.seed_iterator = None
            self.status = None
            return generate_nan_observation(self.observation_space)

    def step(self, action: ActType, **kwargs) -> tuple[ObsType, float, bool, InfoDict]:
        if self.seed_iterator is None:
            raise RuntimeError('State queue is already exhausted, but the environment is still receiving action.')

        # Action is what we have got from policy
        self.status.action_history.append(action)
        action = self.action_interpreter(action, self.simulator.get_state())

        # Use the converted action of update the simulator
        self.simulator.step(action)

        # Update "done" first, as this status might be used by reward_fn later
        done = self.simulator.done()
        self.status.done = done

        # Get state and calculate observation
        sim_state = self.simulator.get_state()
        obs = self.state_interpreter(sim_state)
        self.status.obs_history.append(obs)

        # Reward and extra info
        if self.reward_fn is not None:
            rew = self.reward_fn(sim_state)
        else:
            rew = 0.
        self.status.reward_history.append(rew)

        if self.log_collector is not None:
            log = self.log_collector(sim_state)
        else:
            log = {}

        if self.aux_info_collector is not None:
            aux_info = self.aux_info_collector(sim_state)
        else:
            aux_info = {}

        return obs, rew, done, InfoDict(log=log, aux_info=aux_info)

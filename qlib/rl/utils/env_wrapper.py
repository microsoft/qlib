# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import weakref
from typing import Callable, Any, Iterable, Iterator, TypedDict, Generic, cast

import gym

from qlib.rl.aux_info import AuxiliaryInfoCollector
from qlib.rl.simulator import Simulator, InitialStateType, StateType, ActType
from qlib.rl.interpreter import StateInterpreter, ActionInterpreter, PolicyActType, ObsType
from qlib.rl.reward import Reward

from .finite_env import generate_nan_observation
from .log import LogCollector, LogLevel

# in this case, there won't be any seed for simulator
SEED_INTERATOR_MISSING = "_missing_"

class InfoDict(TypedDict):
    aux_info: dict          # Any information depends on auxiliary info collector
    log: dict[str, Any]     # collected by LogCollector


class EnvWrapperStatus(TypedDict):
    """
    This is the status data structure used in EnvWrapper.
    The fields here are in the semantics of RL.
    For example, ``obs`` means the observation fed into policy.
    ``action`` means the raw action returned by policy.
    """
    cur_step: int
    done: bool
    initial_state: Any | None
    obs_history: list
    action_history: list
    reward_history: list


class EnvWrapper(
    gym.Env[ObsType, PolicyActType],
    Generic[InitialStateType, StateType, ActType, ObsType, PolicyActType]
):
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

    simulator: Simulator[InitialStateType, StateType, ActType]
    seed_iterator: str | Iterator[InitialStateType] | None

    def __init__(
        self,
        simulator_fn: Callable[..., Simulator[InitialStateType, StateType, ActType]],
        state_interpreter: StateInterpreter[StateType, ObsType],
        action_interpreter: ActionInterpreter[StateType, PolicyActType, ActType],
        seed_iterator: Iterable[InitialStateType] | None,
        reward_fn: Reward | None = None,
        aux_info_collector: AuxiliaryInfoCollector[StateType, Any] | None = None,
        logger: LogCollector | None = None
    ):
        # assign weak reference to wrapper
        for obj in [state_interpreter, action_interpreter, reward_fn, aux_info_collector]:
            if obj is not None:
                obj.env = weakref.proxy(self)  # type: ignore

        self.simulator_fn = simulator_fn
        self.state_interpreter = state_interpreter
        self.action_interpreter = action_interpreter

        if seed_iterator is None:
            # in this case, there won't be any seed for simulator
            self.seed_iterator = SEED_INTERATOR_MISSING
        else:
            self.seed_iterator = iter(seed_iterator)
        self.reward_fn = reward_fn

        self.aux_info_collector = aux_info_collector
        self.logger: LogCollector = logger or LogCollector()
        self.status: EnvWrapperStatus = cast(EnvWrapperStatus, None)

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
                raise RuntimeError("You can trying to get a state from a dead environment wrapper.")

            # TODO: simulator/observation might need seed to prefetch something
            # as only seed has the ability to do the work beforehands

            # NOTE: though logger is reset here, logs in this function won't work,
            # because we can't send them outside.
            # See https://github.com/thu-ml/tianshou/issues/605
            self.logger.reset()

            if self.seed_iterator is SEED_INTERATOR_MISSING:
                # no initial state
                self.simulator = cast(Callable[[], Simulator], self.simulator_fn)()
            else:
                initial_state = next(cast(Iterator[InitialStateType], self.seed_iterator))
                self.simulator = self.simulator_fn(initial_state)

            self.simulator.env = cast(EnvWrapper, weakref.proxy(self))

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
            return generate_nan_observation(self.observation_space)

    def step(self, policy_action: PolicyActType, **kwargs) -> tuple[ObsType, float, bool, InfoDict]:
        if self.seed_iterator is None:
            raise RuntimeError("State queue is already exhausted, but the environment is still receiving action.")

        # Clear the logged information from last step
        self.logger.reset()

        self.status["cur_step"] += 1

        # Action is what we have got from policy
        self.status["action_history"].append(policy_action)
        action = self.action_interpreter(self.simulator.get_state(), policy_action)

        # Use the converted action of update the simulator
        self.simulator.step(action)

        # Update "done" first, as this status might be used by reward_fn later
        done = self.simulator.done()
        self.status["done"] = done

        # Get state and calculate observation
        sim_state = self.simulator.get_state()
        obs = self.state_interpreter(sim_state)
        self.status["obs_history"].append(obs)

        # Reward and extra info
        if self.reward_fn is not None:
            rew = self.reward_fn(sim_state)
        else:
            # No reward. Treated as 0.
            rew = 0.
        self.status["reward_history"].append(rew)

        if self.aux_info_collector is not None:
            aux_info = self.aux_info_collector(sim_state)
        else:
            aux_info = {}

        # Final logging stuff: RL-specific logs
        if done:
            self.logger.add_scalar("steps_per_episode", self.status["cur_step"])
        self.logger.add_scalar("reward", rew)
        self.logger.add_any("obs", obs, loglevel=LogLevel.DEBUG)
        self.logger.add_any("policy_act", policy_action, loglevel=LogLevel.DEBUG)

        info_dict = InfoDict(log=self.logger.logs(), aux_info=aux_info)
        return obs, rew, done, info_dict

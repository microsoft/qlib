import gym
import time
import torch
import warnings
import numpy as np
from copy import deepcopy
from numbers import Number
from typing import Any, Dict, List, Union, Optional, Callable

from vecenv import BaseVectorEnv
from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer, ListReplayBuffer, to_numpy
from tianshou.exploration import BaseNoise
from tianshou.env import DummyVectorEnv
from tianshou.data.collector import _batch_set_item


class Collector(object):
    def __init__(
        self,
        policy: BasePolicy,
        env: Union[gym.Env, BaseVectorEnv],
        testing=False,
        buffer: Optional[ReplayBuffer] = None,
        preprocess_fn: Optional[Callable[..., Batch]] = None,
        action_noise: Optional[BaseNoise] = None,
        reward_metric: Optional[Callable[[np.ndarray], float]] = np.sum,
    ) -> None:
        super().__init__()
        if not isinstance(env, BaseVectorEnv):
            env = DummyVectorEnv([lambda: env])
        self.env = env
        self.env_num = len(env)
        # environments that are available in step()
        # this means all environments in synchronous simulation
        # but only a subset of environments in asynchronous simulation
        self._ready_env_ids = np.arange(self.env_num)
        # self.async is a flag to indicate whether this collector works
        # with asynchronous simulation
        self.is_async = env.is_async
        self.testing = testing
        # need cache buffers before storing in the main buffer
        self._cached_buf = [ListReplayBuffer() for _ in range(self.env_num)]
        self.buffer = buffer
        self.policy = policy
        self.preprocess_fn = preprocess_fn
        self.process_fn = policy.process_fn
        # self._action_space = env.action_space
        self._action_noise = action_noise
        self._rew_metric = reward_metric or Collector._default_rew_metric
        # avoid creating attribute outside __init__
        # self.reset()

    @staticmethod
    def _default_rew_metric(x: Union[Number, np.number]) -> Union[Number, np.number]:
        # this internal function is designed for single-agent RL
        # for multi-agent RL, a reward_metric must be provided
        assert np.asanyarray(x).size == 1, "Please specify the reward_metric " "since the reward is not a scalar."
        return x

    def reset(self) -> None:
        """Reset all related variables in the collector."""
        # use empty Batch for ``state`` so that ``self.data`` supports slicing
        # convert empty Batch to None when passing data to policy
        self.data = Batch(state={}, obs={}, act={}, rew={}, done={}, info={}, obs_next={}, policy={})
        self.reset_env()
        self.reset_buffer()
        self.reset_stat()
        if self._action_noise is not None:
            self._action_noise.reset()

    def reset_stat(self) -> None:
        """Reset the statistic variables."""
        self.collect_time, self.collect_step, self.collect_episode = 0.0, 0, 0

    def reset_buffer(self) -> None:
        """Reset the main data buffer."""
        if self.buffer is not None:
            self.buffer.reset()

    def get_env_num(self) -> int:
        """ """
        return self.env_num

    def reset_env(self) -> None:
        """Reset all of the environment(s)' states and the cache buffers."""
        self._ready_env_ids = np.arange(self.env_num)
        self.env.reset_sampler()
        obs, stop_id = self.env.reset()
        if self.preprocess_fn:
            obs = self.preprocess_fn(obs=obs).get("obs", obs)
        self.data.obs = obs
        for b in self._cached_buf:
            b.reset()
        self._ready_env_ids = np.array([x for x in self._ready_env_ids if x not in stop_id])

    def _reset_state(self, id: Union[int, List[int]]) -> None:
        """Reset the hidden state: self.data.state[id]."""
        state = self.data.state  # it is a reference
        if isinstance(state, torch.Tensor):
            state[id].zero_()
        elif isinstance(state, np.ndarray):
            state[id] = None if state.dtype == np.object else 0
        elif isinstance(state, Batch):
            state.empty_(id)

    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[Union[int, List[int]]] = None,
        random: bool = False,
        render: Optional[float] = None,
        log_fn=None,
        no_grad: bool = True,
    ) -> Dict[str, float]:
        """Collect a specified number of step or episode.

        :param int: n_step: how many steps you want to collect.
        :param n_episode: how many episodes you want to collect. If it is an
            int, it means to collect at lease ``n_episode`` episodes; if it is
            a list, it means to collect exactly ``n_episode[i]`` episodes in
            the i-th environment
        :param bool: random: whether to use random policy for collecting data,
            defaults to False.
        :param float: render: the sleep time between rendering consecutive
            frames, defaults to None (no rendering).
        :param bool: no_grad: whether to retain gradient in policy.forward,
            defaults to True (no gradient retaining).

        .. note::

        One and only one collection number specification is permitted,
        either ``n_step`` or ``n_episode``.

        :param n_step: Optional[int]:  (Default value = None)
        :param n_episode: Optional[Union[int:List[int]]]:  (Default value = None)
        :param random: bool:  (Default value = False)
        :param render: Optional[float]:  (Default value = None)
        :param log_fn: Default value = None)
        :param no_grad: bool:  (Default value = True)
        :param n_step: Optional[int]:  (Default value = None)
        :param n_episode: Optional[Union[int:
        :param List[int]]]: (Default value = None)
        :param random: bool:  (Default value = False)
        :param render: Optional[float]:  (Default value = None)
        :param no_grad: bool:  (Default value = True)
        :param n_step: Optional[int]:  (Default value = None)
        :param n_episode: Optional[Union[int:
        :param random: bool:  (Default value = False)
        :param render: Optional[float]:  (Default value = None)
        :param no_grad: bool:  (Default value = True)
        :returns: A dict including the following keys

          * ``n/ep`` the collected number of episodes.
          * ``n/st`` the collected number of steps.
          * ``v/st`` the speed of steps per second.
          * ``v/ep`` the speed of episode per second.
          * ``rew`` the mean reward over collected episodes.
          * ``len`` the mean length over collected episodes.

        """
        assert (
            (n_step is not None and n_episode is None and n_step > 0)
            or (n_step is None and n_episode is not None and np.sum(n_episode) > 0)
            or self.testing
        ), "Only one of n_step or n_episode is allowed in Collector.collect, "
        f"got n_step = {n_step}, n_episode = {n_episode}."
        start_time = time.time()
        step_count = 0
        step_time = 0.0
        reset_time = 0.0
        model_time = 0.0
        # episode of each environment
        episode_count = np.zeros(self.env_num)
        # If n_episode is a list, and some envs have collected the required
        # number of episodes, these envs will be recorded in this list, and
        # they will not be stepped.
        finished_env_ids = []
        rewards = []
        whole_data = Batch()
        if isinstance(n_episode, list):
            assert len(n_episode) == self.get_env_num()
            finished_env_ids = [i for i in self._ready_env_ids if n_episode[i] <= 0]
            self._ready_env_ids = np.array([x for x in self._ready_env_ids if x not in finished_env_ids])
        while True:
            if step_count >= 100000 and episode_count.sum() == 0:
                warnings.warn(
                    "There are already many steps in an episode. "
                    "You should add a time limitation to your environment!",
                    Warning,
                )

            is_async = self.is_async or len(finished_env_ids) > 0
            if is_async:
                # self.data are the data for all environments in async
                # simulation or some envs have finished,
                # **only a subset of data are disposed**,
                # so we store the whole data in ``whole_data``, let self.data
                # to be the data available in ready environments, and finally
                # set these back into all the data
                whole_data = self.data
                self.data = self.data[self._ready_env_ids]

            # restore the state and the input data
            last_state = self.data.state
            if isinstance(last_state, Batch) and last_state.is_empty():
                last_state = None
            self.data.update(state=Batch(), obs_next=Batch(), policy=Batch())

            # calculate the next action
            start = time.time()
            if random:
                spaces = self._action_space
                result = Batch(act=[spaces[i].sample() for i in self._ready_env_ids])
            else:
                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
                        result = self.policy(self.data, last_state)
                else:
                    result = self.policy(self.data, last_state)
            model_time += time.time() - start
            state = result.get("state", Batch())
            # convert None to Batch(), since None is reserved for 0-init
            if state is None:
                state = Batch()
            self.data.update(state=state, policy=result.get("policy", Batch()))
            # save hidden state to policy._state, in order to save into buffer
            if not (isinstance(state, Batch) and state.is_empty()):
                self.data.policy._state = self.data.state

            self.data.act = to_numpy(result.act)
            if self._action_noise is not None:
                assert isinstance(self.data.act, np.ndarray)
                self.data.act += self._action_noise(self.data.act.shape)

            # step in env
            start = time.time()
            if not is_async:
                obs_next, rew, done, info = self.env.step(self.data.act)
                if log_fn:
                    log_fn(info)
            else:
                # store computed actions, states, etc
                _batch_set_item(whole_data, self._ready_env_ids, self.data, self.env_num)
                # fetch finished data
                obs_next, rew, done, info = self.env.step(self.data.act, id=self._ready_env_ids)
                self._ready_env_ids = np.array([i["env_id"] for i in info])
                # get the stepped data
                self.data = whole_data[self._ready_env_ids]
                if log_fn:
                    log_fn(info)

            step_time += time.time() - start
            # move data to self.data
            self.data.update(obs_next=obs_next, rew=rew, done=done, info=[{} for i in info])

            if render:
                self.env.render()
                time.sleep(render)

            # add data into the buffer
            if self.preprocess_fn:
                result = self.preprocess_fn(**self.data)  # type: ignore
                self.data.update(result)

            for j, i in enumerate(self._ready_env_ids):
                # j is the index in current ready_env_ids
                # i is the index in all environments
                if self.buffer is None:
                    # users do not want to store data, so we store
                    # small fake data here to make the code clean
                    self._cached_buf[i].add(obs=0, act=0, rew=rew[j], done=0)
                else:
                    self._cached_buf[i].add(**self.data[j])

                if done[j]:
                    if not (isinstance(n_episode, list) and episode_count[i] >= n_episode[i]):
                        episode_count[i] += 1
                        rewards.append(self._rew_metric(np.sum(self._cached_buf[i].rew, axis=0)))
                        step_count += len(self._cached_buf[i])
                        if self.buffer is not None:
                            self.buffer.update(self._cached_buf[i])
                        if isinstance(n_episode, list) and episode_count[i] >= n_episode[i]:
                            # env i has collected enough data, it has finished
                            finished_env_ids.append(i)
                    self._cached_buf[i].reset()
                    self._reset_state(j)
            obs_next = self.data.obs_next
            start = time.time()
            if sum(done):
                env_ind_local = np.where(done)[0].tolist()
                env_ind_global = self._ready_env_ids[env_ind_local]
                obs_reset, stop_id = self.env.reset(env_ind_global)
                _ready_env_ids = self._ready_env_ids.tolist()
                for i in stop_id:
                    finished_env_ids.append(i)
                    # env_ind_local.remove(_ready_env_ids.index(i))
                if len(env_ind_local) > 0:
                    if self.preprocess_fn:
                        obs_reset = self.preprocess_fn(obs=obs_reset).get("obs", obs_reset)
                    obs_next[env_ind_local] = obs_reset
            reset_time += time.time() - start
            self.data.obs = obs_next
            if is_async:
                # set data back
                whole_data = deepcopy(whole_data)  # avoid reference in ListBuf
                _batch_set_item(whole_data, self._ready_env_ids, self.data, self.env_num)
                # let self.data be the data in all environments again
                self.data = whole_data
            self._ready_env_ids = np.array([x for x in self._ready_env_ids if x not in finished_env_ids])
            if n_step:
                if step_count >= n_step:
                    break
            else:
                if isinstance(n_episode, int) and episode_count.sum() >= n_episode:
                    break
                if isinstance(n_episode, list) and (episode_count >= n_episode).all():
                    break
            if len(self._ready_env_ids) == 0 and self.testing:
                break

        # finished envs are ready, and can be used for the next collection
        self._ready_env_ids = np.array(self._ready_env_ids.tolist() + finished_env_ids)

        # generate the statistics
        episode_count = sum(episode_count)
        duration = max(time.time() - start_time, 1e-9)
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += duration
        return {
            "n/ep": episode_count,
            "n/st": step_count,
            "v/st": step_count / duration,
            "v/ep": episode_count / duration,
            "t/st": step_time / step_count,
            "t/re": reset_time / episode_count,
            "t/mo": model_time / step_count,
            "rew": np.mean(rewards),
            "rew_std": np.std(rewards),
            "len": step_count / episode_count,
        }

import gym
import time
import ctypes
import numpy as np
from collections import OrderedDict
from multiprocessing.context import Process
from multiprocessing import Array, Pipe, connection, Queue
from typing import Any, List, Tuple, Union, Callable, Optional

from tianshou.env.worker import EnvWorker
from tianshou.env.utils import CloudpickleWrapper


_NP_TO_CT = {
    np.bool: ctypes.c_bool,
    np.bool_: ctypes.c_bool,
    np.uint8: ctypes.c_uint8,
    np.uint16: ctypes.c_uint16,
    np.uint32: ctypes.c_uint32,
    np.uint64: ctypes.c_uint64,
    np.int8: ctypes.c_int8,
    np.int16: ctypes.c_int16,
    np.int32: ctypes.c_int32,
    np.int64: ctypes.c_int64,
    np.float32: ctypes.c_float,
    np.float64: ctypes.c_double,
}


class ShArray:
    """Wrapper of multiprocessing Array."""

    def __init__(self, dtype: np.generic, shape: Tuple[int]) -> None:
        self.arr = Array(
            _NP_TO_CT[dtype.type],  # type: ignore
            int(np.prod(shape)),
        )
        self.dtype = dtype
        self.shape = shape

    def save(self, ndarray: np.ndarray) -> None:
        """

        :param ndarray: np.ndarray:
        :param ndarray: np.ndarray:
        :param ndarray: np.ndarray:

        """
        assert isinstance(ndarray, np.ndarray)
        dst = self.arr.get_obj()
        dst_np = np.frombuffer(dst, dtype=self.dtype).reshape(self.shape)
        np.copyto(dst_np, ndarray)

    def get(self) -> np.ndarray:
        """ """
        obj = self.arr.get_obj()
        return np.frombuffer(obj, dtype=self.dtype).reshape(self.shape)


def _setup_buf(space: gym.Space) -> Union[dict, tuple, ShArray]:
    """

    :param space: gym.Space:
    :param space: gym.Space:
    :param space: gym.Space:

    """
    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict)
        return {k: _setup_buf(v) for k, v in space.spaces.items()}
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(space.spaces, tuple)
        return tuple([_setup_buf(t) for t in space.spaces])
    else:
        return ShArray(space.dtype, space.shape)


def _worker(
    parent: connection.Connection,
    p: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    obs_bufs: Optional[Union[dict, tuple, ShArray]] = None,
) -> None:
    """

    :param parent: connection.Connection:
    :param p: connection.Connection:
    :param env_fn_wrapper: CloudpickleWrapper:
    :param obs_bufs: Optional[Union[dict:
    :param tuple: param ShArray]]:  (Default value = None)
    :param parent: connection.Connection:
    :param p: connection.Connection:
    :param env_fn_wrapper: CloudpickleWrapper:
    :param obs_bufs: Optional[Union[dict:
    :param ShArray]]: (Default value = None)
    :param parent: connection.Connection:
    :param p: connection.Connection:
    :param env_fn_wrapper: CloudpickleWrapper:
    :param obs_bufs: Optional[Union[dict:

    """

    def _encode_obs(obs: Union[dict, tuple, np.ndarray], buffer: Union[dict, tuple, ShArray],) -> None:
        """

        :param obs: Union[dict:
        :param tuple: param np.ndarray]:
        :param buffer: Union[dict:
        :param ShArray:
        :param obs: Union[dict:
        :param np.ndarray]:
        :param buffer: Union[dict:
        :param ShArray]:
        :param obs: Union[dict:
        :param buffer: Union[dict:

        """
        if isinstance(obs, np.ndarray) and isinstance(buffer, ShArray):
            buffer.save(obs)
        elif isinstance(obs, tuple) and isinstance(buffer, tuple):
            for o, b in zip(obs, buffer):
                _encode_obs(o, b)
        elif isinstance(obs, dict) and isinstance(buffer, dict):
            for k in obs.keys():
                _encode_obs(obs[k], buffer[k])
        return None

    parent.close()
    env = env_fn_wrapper.data()
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:  # the pipe has been closed
                p.close()
                break
            if cmd == "step":
                obs, reward, done, info = env.step(data)
                if obs_bufs is not None:
                    _encode_obs(obs, obs_bufs)
                    obs = None
                p.send((obs, reward, done, info))
            elif cmd == "reset":
                obs = env.reset(data)
                if obs_bufs is not None:
                    _encode_obs(obs, obs_bufs)
                    obs = None
                p.send(obs)
            elif cmd == "close":
                p.send(env.close())
                p.close()
                break
            elif cmd == "render":
                p.send(env.render(**data) if hasattr(env, "render") else None)
            elif cmd == "seed":
                p.send(env.seed(data) if hasattr(env, "seed") else None)
            elif cmd == "getattr":
                p.send(getattr(env, data) if hasattr(env, data) else None)
            elif cmd == "toggle_log":
                env.toggle_log(data)
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class SubprocEnvWorker(EnvWorker):
    """Subprocess worker used in SubprocVectorEnv and ShmemVectorEnv."""

    def __init__(self, env_fn: Callable[[], gym.Env], share_memory: bool = False) -> None:
        super().__init__(env_fn)
        self.parent_remote, self.child_remote = Pipe()
        self.share_memory = share_memory
        self.buffer: Optional[Union[dict, tuple, ShArray]] = None
        if self.share_memory:
            dummy = env_fn()
            obs_space = dummy.observation_space
            dummy.close()
            del dummy
            self.buffer = _setup_buf(obs_space)
        args = (
            self.parent_remote,
            self.child_remote,
            CloudpickleWrapper(env_fn),
            self.buffer,
        )
        self.process = Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()

    def __getattr__(self, key: str) -> Any:
        self.parent_remote.send(["getattr", key])
        return self.parent_remote.recv()

    def _decode_obs(self) -> Union[dict, tuple, np.ndarray]:
        """ """

        def decode_obs(buffer: Optional[Union[dict, tuple, ShArray]]) -> Union[dict, tuple, np.ndarray]:
            """

            :param buffer: Optional[Union[dict:
            :param tuple: param ShArray]]:
            :param buffer: Optional[Union[dict:
            :param ShArray]]:
            :param buffer: Optional[Union[dict:

            """
            if isinstance(buffer, ShArray):
                return buffer.get()
            elif isinstance(buffer, tuple):
                return tuple([decode_obs(b) for b in buffer])
            elif isinstance(buffer, dict):
                return {k: decode_obs(v) for k, v in buffer.items()}
            else:
                raise NotImplementedError

        return decode_obs(self.buffer)

    def reset(self, sample) -> Any:
        """

        :param sample:

        """
        self.parent_remote.send(["reset", sample])
        # obs = self.parent_remote.recv()
        # if self.share_memory:
        #     obs = self._decode_obs()
        # return obs

    def get_reset_result(self):
        """ """
        obs = self.parent_remote.recv()
        if self.share_memory:
            obs = self._decode_obs()
        return obs

    @staticmethod
    def wait(  # type: ignore
        workers: List["SubprocEnvWorker"], wait_num: int, timeout: Optional[float] = None,
    ) -> List["SubprocEnvWorker"]:
        """

        :param # type: ignoreworkers: List["SubprocEnvWorker"]:
        :param wait_num: int:
        :param timeout: Optional[float]:  (Default value = None)
        :param # type: ignoreworkers: List["SubprocEnvWorker"]:
        :param wait_num: int:
        :param timeout: Optional[float]:  (Default value = None)

        """
        remain_conns = conns = [x.parent_remote for x in workers]
        ready_conns: List[connection.Connection] = []
        remain_time, t1 = timeout, time.time()
        while len(remain_conns) > 0 and len(ready_conns) < wait_num:
            if timeout:
                remain_time = timeout - (time.time() - t1)
                if remain_time <= 0:
                    break
            # connection.wait hangs if the list is empty
            new_ready_conns = connection.wait(remain_conns, timeout=remain_time)
            ready_conns.extend(new_ready_conns)  # type: ignore
            remain_conns = [conn for conn in remain_conns if conn not in ready_conns]
        return [workers[conns.index(con)] for con in ready_conns]

    def send_action(self, action: np.ndarray) -> None:
        """

        :param action: np.ndarray:
        :param action: np.ndarray:
        :param action: np.ndarray:

        """
        self.parent_remote.send(["step", action])

    def toggle_log(self, log):
        self.parent_remote.send(["toggle_log", log])

    def get_result(self,) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ """
        obs, rew, done, info = self.parent_remote.recv()
        if self.share_memory:
            obs = self._decode_obs()
        return obs, rew, done, info

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        """

        :param seed: Optional[int]:  (Default value = None)
        :param seed: Optional[int]:  (Default value = None)
        :param seed: Optional[int]:  (Default value = None)

        """
        self.parent_remote.send(["seed", seed])
        return self.parent_remote.recv()

    def render(self, **kwargs: Any) -> Any:
        """

        :param **kwargs: Any:
        :param **kwargs: Any:

        """
        self.parent_remote.send(["render", kwargs])
        return self.parent_remote.recv()

    def close_env(self) -> None:
        """ """
        try:
            self.parent_remote.send(["close", None])
            # mp may be deleted so it may raise AttributeError
            self.parent_remote.recv()
            self.process.join()
        except (BrokenPipeError, EOFError, AttributeError):
            pass
        # ensure the subproc is terminated
        self.process.terminate()


class BaseVectorEnv(gym.Env):
    """Base class for vectorized environments wrapper.
    Usage:
    ::
        env_num = 8
        envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(env_num)])
        assert len(envs) == env_num
    It accepts a list of environment generators. In other words, an environment
    generator ``efn`` of a specific task means that ``efn()`` returns the
    environment of the given task, for example, ``gym.make(task)``.
    All of the VectorEnv must inherit :class:`~tianshou.env.BaseVectorEnv`.
    Here are some other usages:
    ::
        envs.seed(2)  # which is equal to the next line
        envs.seed([2, 3, 4, 5, 6, 7, 8, 9])  # set specific seed for each env
        obs = envs.reset()  # reset all environments
        obs = envs.reset([0, 5, 7])  # reset 3 specific environments
        obs, rew, done, info = envs.step([1] * 8)  # step synchronously
        envs.render()  # render all environments
        envs.close()  # close all environments
    .. warning::
        If you use your own environment, please make sure the ``seed`` method
        is set up properly, e.g.,
        ::
            def seed(self, seed):
                np.random.seed(seed)
        Otherwise, the outputs of these envs may be the same with each other.

    :param env_fns: a list of callable envs
    :param env:
    :param worker_fn: a callable worker
    :param worker: which contains the i
    :param int: wait_num
    :param env: step
    :param environments: to finish a step is time
    :param return: when
    :param simulation: in these environments
    :param is: disabled
    :param float: timeout
    :param vectorized: step it only deal with those environments spending time
    :param within: timeout

    """

    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        worker_fn: Callable[[Callable[[], gym.Env]], EnvWorker],
        sampler=None,
        testing: Optional[bool] = False,
        wait_num: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._env_fns = env_fns
        # A VectorEnv contains a pool of EnvWorkers, which corresponds to
        # interact with the given envs (one worker <-> one env).
        self.workers = [worker_fn(fn) for fn in env_fns]
        self.worker_class = type(self.workers[0])
        assert issubclass(self.worker_class, EnvWorker)
        assert all([isinstance(w, self.worker_class) for w in self.workers])

        self.env_num = len(env_fns)
        self.wait_num = wait_num or len(env_fns)
        assert 1 <= self.wait_num <= len(env_fns), f"wait_num should be in [1, {len(env_fns)}], but got {wait_num}"
        self.timeout = timeout
        assert self.timeout is None or self.timeout > 0, f"timeout is {timeout}, it should be positive if provided!"
        self.is_async = self.wait_num != len(env_fns) or timeout is not None or testing
        self.waiting_conn: List[EnvWorker] = []
        # environments in self.ready_id is actually ready
        # but environments in self.waiting_id are just waiting when checked,
        # and they may be ready now, but this is not known until we check it
        # in the step() function
        self.waiting_id: List[int] = []
        # all environments are ready in the beginning
        self.ready_id = list(range(self.env_num))
        self.is_closed = False
        self.sampler = sampler
        self.sample_obs = None

    def _assert_is_not_closed(self) -> None:
        """ """
        assert not self.is_closed, f"Methods of {self.__class__.__name__} cannot be called after " "close."

    def __len__(self) -> int:
        """Return len(self), which is the number of environments."""
        return self.env_num

    def __getattribute__(self, key: str) -> Any:
        """Switch the attribute getter depending on the key.
        Any class who inherits ``gym.Env`` will inherit some attributes, like
        ``action_space``. However, we would like the attribute lookup to go
        straight into the worker (in fact, this vector env's action_space is
        always None).
        """
        if key in [
            "metadata",
            "reward_range",
            "spec",
            "action_space",
            "observation_space",
        ]:  # reserved keys in gym.Env
            return self.__getattr__(key)
        else:
            return super().__getattribute__(key)

    def __getattr__(self, key: str) -> List[Any]:
        """Fetch a list of env attributes.
        This function tries to retrieve an attribute from each individual
        wrapped environment, if it does not belong to the wrapping vector
        environment class.
        """
        return [getattr(worker, key) for worker in self.workers]

    def _wrap_id(self, id: Optional[Union[int, List[int], np.ndarray]] = None) -> Union[List[int], np.ndarray]:
        """

        :param id: Optional[Union[int:
        :param List: int]:
        :param np: ndarray]]:  (Default value = None)
        :param id: Optional[Union[int:
        :param List[int]:
        :param np.ndarray]]: (Default value = None)
        :param id: Optional[Union[int:

        """
        if id is None:
            id = list(range(self.env_num))
        elif np.isscalar(id):
            id = [id]
        return id

    def _assert_id(self, id: List[int]) -> None:
        """

        :param id: List[int]:
        :param id: List[int]:
        :param id: List[int]:

        """
        for i in id:
            assert i not in self.waiting_id, f"Cannot interact with environment {i} which is stepping now."
            assert i in self.ready_id, f"Can only interact with ready environments {self.ready_id}."

    def reset(self, id: Optional[Union[int, List[int], np.ndarray]] = None) -> np.ndarray:
        """Reset the state of some envs and return initial observations.
        If id is None, reset the state of all the environments and return
        initial observations, otherwise reset the specific environments with
        the given id, either an int or a list.

        :param id: Optional[Union[int:
        :param List: int]:
        :param np: ndarray]]:  (Default value = None)
        :param id: Optional[Union[int:
        :param List[int]:
        :param np.ndarray]]: (Default value = None)
        :param id: Optional[Union[int:

        """
        start_time = time.time()
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)
        obs = []
        stop_id = []
        for i in id:
            sample = self.sampler.sample()
            if sample is None:
                stop_id.append(i)
            else:
                self.workers[i].reset(sample)
        for i in id:
            if i in stop_id:
                obs.append(self.sample_obs)
            else:
                this_obs = self.workers[i].get_reset_result()
                if self.sample_obs is None:
                    self.sample_obs = this_obs
                    for j in range(len(obs)):
                        if obs[j] is None:
                            obs[j] = self.sample_obs
                obs.append(this_obs)

        if len(obs) > 0:
            obs = np.stack(obs)
            # if len(stop_id)> 0:
            #     obs_zero =
        # print(time.time() - start_timed)

        return obs, stop_id

    def toggle_log(self, log):
        for worker in self.workers:
            worker.toggle_log(log)

    def reset_sampler(self):
        """ """
        self.sampler.reset()

    def step(self, action: np.ndarray, id: Optional[Union[int, List[int], np.ndarray]] = None) -> List[np.ndarray]:
        """Run one timestep of some environments' dynamics.
        If id is None, run one timestep of all the environments’ dynamics;
        otherwise run one timestep for some environments with given id,  either
        an int or a list. When the end of episode is reached, you are
        responsible for calling reset(id) to reset this environment’s state.
        Accept a batch of action and return a tuple (batch_obs, batch_rew,
        batch_done, batch_info) in numpy format.

        :param numpy: ndarray action: a batch of action provided by the agent.
        :param action: np.ndarray:
        :param id: Optional[Union[int:
        :param List: int]:
        :param np: ndarray]]:  (Default value = None)
        :param action: np.ndarray:
        :param id: Optional[Union[int:
        :param List[int]:
        :param np.ndarray]]: (Default value = None)
        :param action: np.ndarray:
        :param id: Optional[Union[int:
        :rtype: A tuple including four items

        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if not self.is_async:
            assert len(action) == len(id)
            for i, j in enumerate(id):
                self.workers[j].send_action(action[i])
            result = []
            for j in id:
                obs, rew, done, info = self.workers[j].get_result()
                info["env_id"] = j
                result.append((obs, rew, done, info))
        else:
            if action is not None:
                self._assert_id(id)
                assert len(action) == len(id)
                for i, (act, env_id) in enumerate(zip(action, id)):
                    self.workers[env_id].send_action(act)
                    self.waiting_conn.append(self.workers[env_id])
                    self.waiting_id.append(env_id)
                self.ready_id = [x for x in self.ready_id if x not in id]
            ready_conns: List[EnvWorker] = []
            while not ready_conns:
                ready_conns = self.worker_class.wait(self.waiting_conn, self.wait_num, self.timeout)
            result = []
            for conn in ready_conns:
                waiting_index = self.waiting_conn.index(conn)
                self.waiting_conn.pop(waiting_index)
                env_id = self.waiting_id.pop(waiting_index)
                obs, rew, done, info = conn.get_result()
                info["env_id"] = env_id
                result.append((obs, rew, done, info))
                self.ready_id.append(env_id)
        return list(map(np.stack, zip(*result)))

    def seed(self, seed: Optional[Union[int, List[int]]] = None) -> List[Optional[List[int]]]:
        """Set the seed for all environments.
        Accept ``None``, an int (which will extend ``i`` to
        ``[i, i + 1, i + 2, ...]``) or a list.

        :param seed: Optional[Union[int:
        :param List: int]]]:  (Default value = None)
        :param seed: Optional[Union[int:
        :param List[int]]]: (Default value = None)
        :param seed: Optional[Union[int:
        :returns: The list of seeds used in this env's random number generators.
          The first value in the list should be the "main" seed, or the value
          which a reproducer pass to "seed".

        """
        self._assert_is_not_closed()
        seed_list: Union[List[None], List[int]]
        if seed is None:
            seed_list = [seed] * self.env_num
        elif isinstance(seed, int):
            seed_list = [seed + i for i in range(self.env_num)]
        else:
            seed_list = seed
        return [w.seed(s) for w, s in zip(self.workers, seed_list)]

    def render(self, **kwargs: Any) -> List[Any]:
        """Render all of the environments.

        :param **kwargs: Any:
        :param **kwargs: Any:

        """
        self._assert_is_not_closed()
        if self.is_async and len(self.waiting_id) > 0:
            raise RuntimeError(f"Environments {self.waiting_id} are still stepping, cannot " "render them now.")
        return [w.render(**kwargs) for w in self.workers]

    def close(self) -> None:
        """Close all of the environments.
        This function will be called only once (if not, it will be called
        during garbage collected). This way, ``close`` of all workers can be
        assured.


        """
        self._assert_is_not_closed()
        for w in self.workers:
            w.close()
        self.is_closed = True

    def __del__(self) -> None:
        """Redirect to self.close()."""
        if not self.is_closed:
            self.close()


class SubprocVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on subprocess.
    .. seealso::
        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.


    """

    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        sampler=None,
        testing=False,
        wait_num: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        def worker_fn(fn: Callable[[], gym.Env]) -> SubprocEnvWorker:
            """

            :param fn: Callable[[]:
            :param gym: Env]:
            :param fn: Callable[[]:
            :param gym.Env]:
            :param fn: Callable[[]:

            """
            return SubprocEnvWorker(fn, share_memory=False)

        super().__init__(env_fns, worker_fn, sampler, testing, wait_num=wait_num, timeout=timeout)


class ShmemVectorEnv(BaseVectorEnv):
    """Optimized SubprocVectorEnv with shared buffers to exchange observations.
    ShmemVectorEnv has exactly the same API as SubprocVectorEnv.
    .. seealso::
        Please refer to :class:`~tianshou.env.SubprocVectorEnv` for more
        detailed explanation.


    """

    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        sampler=None,
        testing=False,
        wait_num: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        def worker_fn(fn: Callable[[], gym.Env]) -> SubprocEnvWorker:
            """

            :param fn: Callable[[]:
            :param gym: Env]:
            :param fn: Callable[[]:
            :param gym.Env]:
            :param fn: Callable[[]:

            """
            return SubprocEnvWorker(fn, share_memory=True)

        super().__init__(env_fns, worker_fn, sampler, testing, wait_num=wait_num, timeout=timeout)

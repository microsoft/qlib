# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
from threading import Thread
from typing import Callable, Text, Union

from joblib import Parallel, delayed
from joblib._parallel_backends import MultiprocessingBackend
import pandas as pd

from queue import Queue
import concurrent

from qlib.config import C, QlibConfig


class ParallelExt(Parallel):
    def __init__(self, *args, **kwargs):
        maxtasksperchild = kwargs.pop("maxtasksperchild", None)
        super(ParallelExt, self).__init__(*args, **kwargs)
        if isinstance(self._backend, MultiprocessingBackend):
            self._backend_args["maxtasksperchild"] = maxtasksperchild


def datetime_groupby_apply(
    df, apply_func: Union[Callable, Text], axis=0, level="datetime", resample_rule="M", n_jobs=-1
):
    """datetime_groupby_apply
    This function will apply the `apply_func` on the datetime level index.

    Parameters
    ----------
    df :
        DataFrame for processing
    apply_func : Union[Callable, Text]
        apply_func for processing the data
        if a string is given, then it is treated as naive pandas function
    axis :
        which axis is the datetime level located
    level :
        which level is the datetime level
    resample_rule :
        How to resample the data to calculating parallel
    n_jobs :
        n_jobs for joblib
    Returns:
        pd.DataFrame
    """

    def _naive_group_apply(df):
        if isinstance(apply_func, str):
            return getattr(df.groupby(axis=axis, level=level), apply_func)()
        return df.groupby(axis=axis, level=level).apply(apply_func)

    if n_jobs != 1:
        dfs = ParallelExt(n_jobs=n_jobs)(
            delayed(_naive_group_apply)(sub_df) for idx, sub_df in df.resample(resample_rule, axis=axis, level=level)
        )
        return pd.concat(dfs, axis=axis).sort_index()
    else:
        return _naive_group_apply(df)


class AsyncCaller:
    """
    This AsyncCaller tries to make it easier to async call

    Currently, it is used in MLflowRecorder to make functions like `log_params` async

    NOTE:
    - This caller didn't consider the return value
    """

    STOP_MARK = "__STOP"

    def __init__(self) -> None:
        self._q = Queue()
        self._stop = False
        self._t = Thread(target=self.run)
        self._t.start()

    def close(self):
        self._q.put(self.STOP_MARK)

    def run(self):
        while True:
            data = self._q.get()
            if data == self.STOP_MARK:
                break
            data()

    def __call__(self, func, *args, **kwargs):
        self._q.put(partial(func, *args, **kwargs))

    def wait(self, close=True):
        if close:
            self.close()
        self._t.join()

    @staticmethod
    def async_dec(ac_attr):
        def decorator_func(func):
            def wrapper(self, *args, **kwargs):
                if isinstance(getattr(self, ac_attr, None), Callable):
                    return getattr(self, ac_attr)(func, self, *args, **kwargs)
                else:
                    return func(self, *args, **kwargs)

            return wrapper

        return decorator_func


# # Outlines: Joblib enhancement
# The code are for implementing following workflow
# - Construct complex data structure nested with delayed joblib tasks
#      - For example,  {"job": [<delayed_joblib_task>,  {"1": <delayed_joblib_task>}]}
# - executing all the tasks and replace all the <delayed_joblib_task> with its return value

# This will make it easier to convert some existing code to a parallel one


class DelayedTask:
    def get_delayed_tuple(self):
        """get_delayed_tuple.
        Return the delayed_tuple created by joblib.delayed
        """
        raise NotImplementedError("NotImplemented")

    def set_res(self, res):
        """set_res.

        Parameters
        ----------
        res :
            the executed result of the delayed tuple
        """
        self.res = res

    def get_replacement(self):
        """return the object to replace the delayed task"""
        raise NotImplementedError("NotImplemented")


class DelayedTuple(DelayedTask):
    def __init__(self, delayed_tpl):
        self.delayed_tpl = delayed_tpl
        self.res = None

    def get_delayed_tuple(self):
        return self.delayed_tpl

    def get_replacement(self):
        return self.res


class DelayedDict(DelayedTask):
    """DelayedDict.
    It is designed for following feature:
    Converting following existing code to parallel
    - constructing a dict
    - key can be gotten instantly
    - computation of values tasks a lot of time.
        - AND ALL the values are calculated in a SINGLE function
    """

    def __init__(self, key_l, delayed_tpl):
        self.key_l = key_l
        self.delayed_tpl = delayed_tpl

    def get_delayed_tuple(self):
        return self.delayed_tpl

    def get_replacement(self):
        return dict(zip(self.key_l, self.res))


def is_delayed_tuple(obj) -> bool:
    """is_delayed_tuple.

    Parameters
    ----------
    obj : object

    Returns
    -------
    bool
        is `obj` joblib.delayed tuple
    """
    return isinstance(obj, tuple) and len(obj) == 3 and callable(obj[0])


def _replace_and_get_dt(complex_iter):
    """_replace_and_get_dt.

    FIXME: this function may cause infinite loop when the complex data-structure contains loop-reference

    Parameters
    ----------
    complex_iter :
        complex_iter
    """
    if isinstance(complex_iter, DelayedTask):
        dt = complex_iter
        return dt, [dt]
    elif is_delayed_tuple(complex_iter):
        dt = DelayedTuple(complex_iter)
        return dt, [dt]
    elif isinstance(complex_iter, (list, tuple)):
        new_ci = []
        dt_all = []
        for item in complex_iter:
            new_item, dt_list = _replace_and_get_dt(item)
            new_ci.append(new_item)
            dt_all += dt_list
        return new_ci, dt_all
    elif isinstance(complex_iter, dict):
        new_ci = {}
        dt_all = []
        for key, item in complex_iter.items():
            new_item, dt_list = _replace_and_get_dt(item)
            new_ci[key] = new_item
            dt_all += dt_list
        return new_ci, dt_all
    else:
        return complex_iter, []


def _recover_dt(complex_iter):
    """_recover_dt.

    replace all the DelayedTask in the `complex_iter` with its `.res` value

    FIXME: this function may cause infinite loop when the complex data-structure contains loop-reference

    Parameters
    ----------
    complex_iter :
        complex_iter
    """
    if isinstance(complex_iter, DelayedTask):
        return complex_iter.get_replacement()
    elif isinstance(complex_iter, (list, tuple)):
        return [_recover_dt(item) for item in complex_iter]
    elif isinstance(complex_iter, dict):
        return {key: _recover_dt(item) for key, item in complex_iter.items()}
    else:
        return complex_iter


def complex_parallel(paral: Parallel, complex_iter):
    """complex_parallel.
    Find all the delayed function created by delayed in complex_iter, run them parallelly and then replace it with the result

    >>> from qlib.utils.paral import complex_parallel
    >>> from joblib import Parallel, delayed
    >>> complex_iter = {"a": delayed(sum)([1,2,3]), "b": [1, 2, delayed(sum)([10, 1])]}
    >>> complex_parallel(Parallel(), complex_iter)
    {'a': 6, 'b': [1, 2, 11]}

    Parameters
    ----------
    paral : Parallel
        paral
    complex_iter :
        NOTE: only list, tuple and dict will be explored!!!!

    Returns
    -------
    complex_iter whose delayed joblib tasks are replaced with its execution results.
    """

    complex_iter, dt_all = _replace_and_get_dt(complex_iter)
    for res, dt in zip(paral(dt.get_delayed_tuple() for dt in dt_all), dt_all):
        dt.set_res(res)
    complex_iter = _recover_dt(complex_iter)
    return complex_iter


class call_in_subproc:
    """
    When we repeatedly run functions, it is hard to avoid memory leakage.
    So we run it in the subprocess to ensure it is OK.

    NOTE: Because local object can't be pickled. So we can't implement it via closure.
          We have to implement it via callable Class
    """

    def __init__(self, func: Callable, qlib_config: QlibConfig = None):
        """
        Parameters
        ----------
        func : Callable
            the function to be wrapped

        qlib_config : QlibConfig
            Qlib config for initialization in subprocess

        Returns
        -------
        Callable
        """
        self.func = func
        self.qlib_config = qlib_config

    def _func_mod(self, *args, **kwargs):
        """Modify the initial function by adding Qlib initialization"""
        if self.qlib_config is not None:
            C.register_from_C(self.qlib_config)
        return self.func(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            return executor.submit(self._func_mod, *args, **kwargs).result()

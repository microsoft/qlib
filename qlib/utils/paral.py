# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
from threading import Thread
from typing import Callable

from joblib import Parallel, delayed
from joblib._parallel_backends import MultiprocessingBackend
import pandas as pd

from queue import Queue


class ParallelExt(Parallel):
    def __init__(self, *args, **kwargs):
        maxtasksperchild = kwargs.pop("maxtasksperchild", None)
        super(ParallelExt, self).__init__(*args, **kwargs)
        if isinstance(self._backend, MultiprocessingBackend):
            self._backend_args["maxtasksperchild"] = maxtasksperchild


def datetime_groupby_apply(df, apply_func, axis=0, level="datetime", resample_rule="M", n_jobs=-1, skip_group=False):
    """datetime_groupby_apply
    This function will apply the `apply_func` on the datetime level index.

    Parameters
    ----------
    df :
        DataFrame for processing
    apply_func :
        apply_func for processing the data
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

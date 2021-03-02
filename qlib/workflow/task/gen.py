# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
this is a task generator
"""
import abc
import copy
import typing
from .utils import TimeAdjuster


class TaskGen(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> typing.List[dict]:
        """
        generate

        Parameters
        ----------
        args, kwargs:
            The info for generating tasks
            Example 1):
                input: a specific task template
                output: rolling version of the tasks
            Example 2):
                input: a specific task template
                output: a set of tasks with different losses

        Returns
        -------
        typing.List[dict]:
            A list of tasks
        """
        pass


class RollingGen(TaskGen):

    ROLL_EX = TimeAdjuster.SHIFT_EX
    ROLL_SD = TimeAdjuster.SHIFT_SD

    def __init__(self, step: int = 40, rtype: str = ROLL_EX):
        """
        Generate tasks for rolling

        Parameters
        ----------
        step : int
            step to rolling
        rtype : str
            rolling type (expanding, rolling)
        """
        self.step = step
        self.rtype = rtype
        self.ta = TimeAdjuster(future=True)  # 为了保证test最后的日期不是None, 所以这边要改一改

        self.test_key = "test"
        self.train_key = "train"

    def __call__(self, task: dict):
        """
        Converting the task into a rolling task

        Parameters
        ----------
        task : dict
            A dict describing a task. For example.

            DEFAULT_TASK = {
                "model": {
                    "class": "LGBModel",
                    "module_path": "qlib.contrib.model.gbdt",
                },
                "dataset": {
                    "class": "DatasetH",
                    "module_path": "qlib.data.dataset",
                    "kwargs": {
                        "handler": {
                            "class": "Alpha158",
                            "module_path": "qlib.contrib.data.handler",
                            "kwargs": data_handler_config,
                        },
                        "segments": {
                            "train": ("2008-01-01", "2014-12-31"),
                            "valid": ("2015-01-01", "2016-12-20"),  # Please avoid leaking the future test data into validation
                            "test": ("2017-01-01", "2020-08-01"),
                        },
                    },
                },
                # You shoud record the data in specific sequence
                # "record": ['SignalRecord', 'SigAnaRecord', 'PortAnaRecord'],
            }
        """
        res = []

        prev_seg = None
        test_end = None
        while True:
            t = copy.deepcopy(task)

            # calculate segments
            if prev_seg is None:
                # First rolling
                # 1) prepare the end porint
                segments = copy.deepcopy(self.ta.align_seg(t["dataset"]["kwargs"]["segments"]))
                test_end = self.ta.max() if segments[self.test_key][1] is None else segments[self.test_key][1]
                # 2) and the init test segments
                test_start_idx = self.ta.align_idx(segments[self.test_key][0])
                segments[self.test_key] = (self.ta.get(test_start_idx), self.ta.get(test_start_idx + self.step - 1))
            else:
                segments = {}
                try:
                    for k, seg in prev_seg.items():
                        # 决定怎么shift
                        if k == self.train_key and self.rtype == self.ROLL_EX:
                            rtype = self.ta.SHIFT_EX
                        else:
                            rtype = self.ta.SHIFT_SD
                        # 整段数据做shift
                        segments[k] = self.ta.shift(seg, step=self.step, rtype=rtype)
                    if segments[self.test_key][0] > test_end:
                        break
                except KeyError:
                    # We reach the end of tasks
                    # No more rolling
                    break

            t["dataset"]["kwargs"]["segments"] = copy.deepcopy(segments)
            prev_seg = segments
            res.append(t)
        return res

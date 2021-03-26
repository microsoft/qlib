# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
this is a task generator
"""
import abc
import copy
import typing
from .utils import TimeAdjuster


def task_generator(tasks, generators) -> list:
    """Use a list of TaskGen and a list of task templates to generate different tasks.

    For examples:

        There are 3 task templates a,b,c and 2 TaskGen A,B. A will generates 2 tasks from a template and B will generates 3 tasks from a template.
        task_generator([a, b, c], [A, B]) will finally generate 3*2*3 = 18 tasks.

    Parameters
    ----------
    tasks : List[dict] or dict
        a list of task templates or a single task
    generators : List[TaskGen] or TaskGen
        a list of TaskGen or a single TaskGen

    Returns
    -------
    list
        a list of tasks
    """

    if isinstance(tasks, dict):
        tasks = [tasks]
    if isinstance(generators, TaskGen):
        generators = [generators]

    # generate gen_task_list
    gen_task_list = []
    for gen in generators:
        new_task_list = []
        for task in tasks:
            new_task_list.extend(gen.generate(task))
        gen_task_list = new_task_list

    return gen_task_list


class TaskGen(metaclass=abc.ABCMeta):
    """
    the base class for generate different tasks

    Example 1:

        input: a specific task template and rolling steps

        output: rolling version of the tasks

    Example 2:

        input: a specific task template and losses list

        output: a set of tasks with different losses

    """

    @abc.abstractmethod
    def generate(self, task: dict) -> typing.List[dict]:
        """
        generate different tasks based on a task template

        Parameters
        ----------
        task: dict
            a task template

        Returns
        -------
        typing.List[dict]:
            A list of tasks
        """
        pass


class RollingGen(TaskGen):
    ROLL_EX = TimeAdjuster.SHIFT_EX  # fixed start date, expanding end date
    ROLL_SD = TimeAdjuster.SHIFT_SD  # fixed segments size, slide it from start date

    def __init__(self, step: int = 40, rtype: str = ROLL_EX):
        """
        Generate tasks for rolling

        Parameters
        ----------
        step : int
            step to rolling
        rtype : str
            rolling type (expanding, sliding)
        """
        self.step = step
        self.rtype = rtype
        # TODO: Ask pengrong to update future date in dataset
        self.ta = TimeAdjuster(future=True)

        self.test_key = "test"
        self.train_key = "train"

    def generate(self, task: dict):
        """
        Converting the task into a rolling task

        Parameters
        ----------
        task : dict
            A dict describing a task. For example.

            .. code-block:: python

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
                                "kwargs": {
                                    "start_time": "2008-01-01",
                                    "end_time": "2020-08-01",
                                    "fit_start_time": "2008-01-01",
                                    "fit_end_time": "2014-12-31",
                                    "instruments": "csi100",
                                },
                            },
                            "segments": {
                                "train": ("2008-01-01", "2014-12-31"),
                                "valid": ("2015-01-01", "2016-12-20"),  # Please avoid leaking the future test data into validation
                                "test": ("2017-01-01", "2020-08-01"),
                            },
                        },
                    },
                    "record": [
                        {
                            "class": "SignalRecord",
                            "module_path": "qlib.workflow.record_temp",
                        },
                    ]
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
                # 1) prepare the end point
                segments = copy.deepcopy(self.ta.align_seg(t["dataset"]["kwargs"]["segments"]))
                test_end = self.ta.last_date() if segments[self.test_key][1] is None else segments[self.test_key][1]
                # 2) and init test segments
                test_start_idx = self.ta.align_idx(segments[self.test_key][0])
                segments[self.test_key] = (self.ta.get(test_start_idx), self.ta.get(test_start_idx + self.step - 1))
            else:
                segments = {}
                try:
                    for k, seg in prev_seg.items():
                        # decide how to shift
                        # expanding only for train data, the segments size of test data and valid data won't change
                        if k == self.train_key and self.rtype == self.ROLL_EX:
                            rtype = self.ta.SHIFT_EX
                        else:
                            rtype = self.ta.SHIFT_SD
                        # shift the segments data
                        segments[k] = self.ta.shift(seg, step=self.step, rtype=rtype)
                    if segments[self.test_key][0] > test_end:
                        break
                except KeyError:
                    # We reach the end of tasks
                    # No more rolling
                    break

            # update segments of this task
            t["dataset"]["kwargs"]["segments"] = copy.deepcopy(segments)
            prev_seg = segments
            res.append(t)
        return res

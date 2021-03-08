# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
this is a task generator
"""
import abc
import copy
import typing
from .utils import TimeAdjuster


def task_generator(*args, **kwargs) -> list:
    """
    Accept the dict of task config and the TaskGen to generate different tasks.
    There is no limit to the number and position of input.
    The key of input will add to task config.

    for example:
        There are 3 task_config(a,b,c) and 2 TaskGen(A,B). A will double the task_config and B will triple.
        task_generator(a_key=a, b_key=b, c_key=c, A, B) will finally generate 3*2*3 = 18 task_config.

    Parameters
    ----------
    args : dict or TaskGen
    kwargs : dict or TaskGen

    Returns
    -------
    gen_task_list : list
        a list of task config after generating
    """
    tasks_list = []
    gen_list = []

    tmp_id = 1
    for task in args:
        if isinstance(task, dict):
            task["task_key"] = tmp_id
            tmp_id += 1
            tasks_list.append(task)
        elif isinstance(task, TaskGen):
            gen_list.append(task)
        else:
            raise NotImplementedError(f"{type(task)} is not supported in task_generator")

    for key, task in kwargs.items():
        if isinstance(task, dict):
            task["task_key"] = key
            tasks_list.append(task)
        elif isinstance(task, TaskGen):
            gen_list.append(task)
        else:
            raise NotImplementedError(f"{type(task)} is not supported in task_generator")

    # generate gen_task_list
    gen_task_list = []
    for gen in gen_list:
        new_task_list = []
        for task in tasks_list:
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
                # 1) prepare the end point
                segments = copy.deepcopy(self.ta.align_seg(t["dataset"]["kwargs"]["segments"]))
                test_end = self.ta.last_date() if segments[self.test_key][1] is None else segments[self.test_key][1]
                # 2) and the init test segments
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

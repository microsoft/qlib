# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import time
from xxlimited import Str
from qlib.utils import init_instance_by_config, flatten_dict, get_cls_kwargs
from qlib.workflow import R
from qlib.workflow.recorder import Recorder
from qlib.workflow.record_temp import SignalRecord
from qlib.workflow.task.manage import TaskManager, run_task
from qlib.data.dataset import Dataset
from qlib.model.base import Model
import socket


def begin_task_train(task_config: dict, experiment_name: str, *args, **kwargs) -> Recorder:
    """
    Begin a task training with starting a recorder and saving the task config.

    Args:
        task_config (dict)
        experiment_name (str)

    Returns:
        Recorder
    """
    with R.start(experiment_name=experiment_name, recorder_name=str(time.time())):
        R.log_params(**flatten_dict(task_config))
        R.save_objects(**{"task": task_config})  # keep the original format and datatype
        R.set_tags(**{"hostname": socket.gethostname(), "train_status": "begin_task_train"})
        recorder: Recorder = R.get_recorder()
    return recorder


def end_task_train(rec: Recorder, experiment_name: str, *args, **kwargs):
    """
    Finished task training with real model fitting and saving.

    Args:
        rec (Recorder): This recorder will be resumed
        experiment_name (str)

    Returns:
        Recorder
    """
    with R.start(experiment_name=experiment_name, recorder_name=rec.info["name"], resume=True):
        task_config = R.load_object("task")
        # model & dataset initiaiton
        model: Model = init_instance_by_config(task_config["model"])
        dataset: Dataset = init_instance_by_config(task_config["dataset"])
        # model training
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})
        # This dataset is saved for online inference. So the concrete data should not be dumped
        dataset.config(dump_all=False, recursive=True)
        R.save_objects(**{"dataset": dataset})
        # generate records: prediction, backtest, and analysis
        records = task_config.get("record", [])
        if isinstance(records, dict):  # prevent only one dict
            records = [records]
        for record in records:
            cls, kwargs = get_cls_kwargs(record, default_module="qlib.workflow.record_temp")
            if cls is SignalRecord:
                rconf = {"model": model, "dataset": dataset, "recorder": rec}
            else:
                rconf = {"recorder": rec}
            r = cls(**kwargs, **rconf)
            r.generate()
        R.set_tags(**{"train_status": "end_task_train"})
    return rec


def task_train(task_config: dict, experiment_name: str) -> Recorder:
    """
    task based training

    Parameters
    ----------
    task_config : dict
        A dict describes a task setting.
    experiment_name: str
        The name of experiment

    Returns
    ----------
    Recorder : The instance of the recorder
    """
    recorder = begin_task_train(task_config, experiment_name)
    recorder = end_task_train(recorder, experiment_name)
    return recorder


class Trainer:
    """
    The trainer which can train a list of model
    """

    def train(self, tasks: list, *args, **kwargs):
        """Given a list of model definition, begin a training and return the models.

        Returns:
            list: a list of models
        """
        raise NotImplementedError(f"Please implement the `train` method.")

    def end_train(self, models, *args, **kwargs):
        """Given a list of models, finished something in the end of training if you need.

        Returns:
            list: a list of models
        """
        pass


class TrainerR(Trainer):
    """Trainer based on (R)ecorder.

    Assumption: models were defined by `task` and the results will saved to `Recorder`
    """

    def __init__(self, experiment_name, train_func=task_train):
        self.experiment_name = experiment_name
        self.train_func = train_func

    def train(self, tasks: list, train_func=None, *args, **kwargs):
        """Given a list of `task`s and return a list of trained Recorder. The order can be guaranteed.

        Args:
            tasks (list): a list of definition based on `task` dict
            train_func (Callable): the train method which need at least `task` and `experiment_name`. None for default.

        Returns:
            list: a list of Recorders
        """
        if train_func is None:
            train_func = self.train_func
        recs = []
        for task in tasks:
            recs.append(train_func(task, self.experiment_name, *args, **kwargs))
        return recs


class TrainerRM(Trainer):
    """Trainer based on (R)ecorder and Task(M)anager

    Assumption: `task` will be saved to TaskManager and `task` will be fetched and trained from TaskManager
    """

    def __init__(self, experiment_name: str, task_pool: str, train_func=task_train):
        self.experiment_name = experiment_name
        self.task_pool = task_pool
        self.train_func = train_func

    def train(
        self,
        tasks: list,
        train_func=None,
        before_status=TaskManager.STATUS_WAITING,
        after_status=TaskManager.STATUS_DONE,
        *args,
        **kwargs,
    ):
        """Given a list of `task`s and return a list of trained Recorder. The order can be guaranteed.

        This method defaults to a single process, but TaskManager offered a great way to parallel training.
        Users can customize their train_func to realize multiple processes or even multiple machines.

        Args:
            tasks (list): a list of definition based on `task` dict
            train_func (Callable): the train method which need at least `task` and `experiment_name`. None for default.

        Returns:
            list: a list of Recorders
        """
        if train_func is None:
            train_func = self.train_func
        tm = TaskManager(task_pool=self.task_pool)
        _id_list = tm.create_task(tasks)  # all tasks will be saved to MongoDB
        run_task(
            train_func,
            self.task_pool,
            experiment_name=self.experiment_name,
            before_status=before_status,
            after_status=after_status,
            *args,
            **kwargs,
        )

        recs = []
        for _id in _id_list:
            recs.append(tm.re_query(_id)["res"])
        return recs


class DelayTrainerR(TrainerR):
    """
    A delayed implementation based on TrainerR, which means `train` method may only do some preparation and `end_train` method can do the real model fitting.

    """

    def __init__(self, experiment_name, train_func=begin_task_train, end_train_func=end_task_train):
        super().__init__(experiment_name, train_func)
        self.end_train_func = end_train_func
        self.recs = []

    def train(self, tasks: list, train_func, *args, **kwargs):
        """
        Same as `train` of TrainerR, the results will be recorded in self.recs

        Args:
            tasks (list): a list of definition based on `task` dict
            train_func (Callable): the train method which need at least `task` and `experiment_name`. None for default.

        Returns:
            list: a list of Recorders
        """
        self.recs = super().train(tasks, train_func=train_func, *args, **kwargs)
        return self.recs

    def end_train(self, recs=None, end_train_func=None):
        """
        Given a list of Recorder and return a list of trained Recorder.
        This class will finished real data loading and model fitting.

        Args:
            recs (list, optional): a list of Recorder, the tasks have been saved to them. Defaults to None for using self.recs.
            end_train_func (Callable, optional): the end_train method which need at least `rec` and `experiment_name`. Defaults to None for using self.end_train_func.

        Returns:
            list: a list of Recorders
        """
        if recs is None:
            recs = copy.deepcopy(self.recs)
            # the models will be only trained once
            self.recs = []
        if end_train_func is None:
            end_train_func = self.end_train_func
        for rec in recs:
            end_train_func(rec)
        return recs


class DelayTrainerRM(TrainerRM):
    """
    A delayed implementation based on TrainerRM, which means `train` method may only do some preparation and `end_train` method can do the real model fitting.

    """

    def __init__(self, experiment_name, task_pool: str, train_func=begin_task_train, end_train_func=end_task_train):
        super().__init__(experiment_name, task_pool, train_func)
        self.end_train_func = end_train_func

    def train(self, tasks: list, train_func=None, *args, **kwargs):
        """
        Same as `train` of TrainerRM, the results will be recorded in self.recs

        Args:
            tasks (list): a list of definition based on `task` dict
            train_func (Callable): the train method which need at least `task` and `experiment_name`. None for default.

        Returns:
            list: a list of Recorders
        """
        return super().train(tasks, train_func=train_func, after_status=TaskManager.STATUS_PART_DONE, *args, **kwargs)

    def end_train(self, recs, end_train_func=None):
        """
        Given a list of Recorder and return a list of trained Recorder.
        This class will finished real data loading and model fitting.

        Args:
            recs (list, optional): a list of Recorder, the tasks have been saved to them. Defaults to None for using self.recs..
            end_train_func (Callable, optional): the end_train method which need at least `rec` and `experiment_name`. Defaults to None for using self.end_train_func.

        Returns:
            list: a list of Recorders
        """

        if end_train_func is None:
            end_train_func = self.end_train_func
        run_task(
            end_train_func,
            self.task_pool,
            experiment_name=self.experiment_name,
            before_status=TaskManager.STATUS_PART_DONE,
        )
        return recs

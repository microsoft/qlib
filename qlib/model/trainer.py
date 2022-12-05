# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
The Trainer will train a list of tasks and return a list of model recorders.
There are two steps in each Trainer including ``train`` (make model recorder) and ``end_train`` (modify model recorder).

This is a concept called ``DelayTrainer``, which can be used in online simulating for parallel training.
In ``DelayTrainer``, the first step is only to save some necessary info to model recorders, and the second step which will be finished in the end can do some concurrent and time-consuming operations such as model fitting.

``Qlib`` offer two kinds of Trainer, ``TrainerR`` is the simplest way and ``TrainerRM`` is based on TaskManager to help manager tasks lifecycle automatically.
"""

import socket
from typing import Callable, List, Optional

from tqdm.auto import tqdm

from qlib.config import C
from qlib.data.dataset import Dataset
from qlib.data.dataset.weight import Reweighter
from qlib.log import get_module_logger
from qlib.model.base import Model
from qlib.utils import (
    auto_filter_kwargs,
    fill_placeholder,
    flatten_dict,
    init_instance_by_config,
)
from qlib.utils.paral import call_in_subproc
from qlib.workflow import R
from qlib.workflow.recorder import Recorder
from qlib.workflow.task.manage import TaskManager, run_task


def _log_task_info(task_config: dict):
    R.log_params(**flatten_dict(task_config))
    R.save_objects(**{"task": task_config})  # keep the original format and datatype
    R.set_tags(**{"hostname": socket.gethostname()})


def _exe_task(task_config: dict):
    rec = R.get_recorder()
    # model & dataset initiation
    model: Model = init_instance_by_config(task_config["model"], accept_types=Model)
    dataset: Dataset = init_instance_by_config(task_config["dataset"], accept_types=Dataset)
    reweighter: Reweighter = task_config.get("reweighter", None)
    # model training
    auto_filter_kwargs(model.fit)(dataset, reweighter=reweighter)
    R.save_objects(**{"params.pkl": model})
    # this dataset is saved for online inference. So the concrete data should not be dumped
    dataset.config(dump_all=False, recursive=True)
    R.save_objects(**{"dataset": dataset})
    # fill placehorder
    placehorder_value = {"<MODEL>": model, "<DATASET>": dataset}
    task_config = fill_placeholder(task_config, placehorder_value)
    # generate records: prediction, backtest, and analysis
    records = task_config.get("record", [])
    if isinstance(records, dict):  # prevent only one dict
        records = [records]
    for record in records:
        # Some recorder require the parameter `model` and `dataset`.
        # try to automatically pass in them to the initialization function
        # to make defining the tasking easier
        r = init_instance_by_config(
            record,
            recorder=rec,
            default_module="qlib.workflow.record_temp",
            try_kwargs={"model": model, "dataset": dataset},
        )
        r.generate()


def begin_task_train(task_config: dict, experiment_name: str, recorder_name: str = None) -> Recorder:
    """
    Begin task training to start a recorder and save the task config.

    Args:
        task_config (dict): the config of a task
        experiment_name (str): the name of experiment
        recorder_name (str): the given name will be the recorder name. None for using rid.

    Returns:
        Recorder: the model recorder
    """
    with R.start(experiment_name=experiment_name, recorder_name=recorder_name):
        _log_task_info(task_config)
        return R.get_recorder()


def end_task_train(rec: Recorder, experiment_name: str) -> Recorder:
    """
    Finish task training with real model fitting and saving.

    Args:
        rec (Recorder): the recorder will be resumed
        experiment_name (str): the name of experiment

    Returns:
        Recorder: the model recorder
    """
    with R.start(experiment_name=experiment_name, recorder_id=rec.info["id"], resume=True):
        task_config = R.load_object("task")
        _exe_task(task_config)
    return rec


def task_train(task_config: dict, experiment_name: str, recorder_name: str = None) -> Recorder:
    """
    Task based training, will be divided into two steps.

    Parameters
    ----------
    task_config : dict
        The config of a task.
    experiment_name: str
        The name of experiment
    recorder_name: str
        The name of recorder

    Returns
    ----------
    Recorder: The instance of the recorder
    """
    with R.start(experiment_name=experiment_name, recorder_name=recorder_name):
        _log_task_info(task_config)
        _exe_task(task_config)
        return R.get_recorder()


class Trainer:
    """
    The trainer can train a list of models.
    There are Trainer and DelayTrainer, which can be distinguished by when it will finish real training.
    """

    def __init__(self):
        self.delay = False

    def train(self, tasks: list, *args, **kwargs) -> list:
        """
        Given a list of task definitions, begin training, and return the models.

        For Trainer, it finishes real training in this method.
        For DelayTrainer, it only does some preparation in this method.

        Args:
            tasks: a list of tasks

        Returns:
            list: a list of models
        """
        raise NotImplementedError(f"Please implement the `train` method.")

    def end_train(self, models: list, *args, **kwargs) -> list:
        """
        Given a list of models, finished something at the end of training if you need.
        The models may be Recorder, txt file, database, and so on.

        For Trainer, it does some finishing touches in this method.
        For DelayTrainer, it finishes real training in this method.

        Args:
            models: a list of models

        Returns:
            list: a list of models
        """
        # do nothing if you finished all work in `train` method
        return models

    def is_delay(self) -> bool:
        """
        If Trainer will delay finishing `end_train`.

        Returns:
            bool: if DelayTrainer
        """
        return self.delay

    def __call__(self, *args, **kwargs) -> list:
        return self.end_train(self.train(*args, **kwargs))

    def has_worker(self) -> bool:
        """
        Some trainer has backend worker to support parallel training
        This method can tell if the worker is enabled.

        Returns
        -------
        bool:
            if the worker is enabled

        """
        return False

    def worker(self):
        """
        start the worker

        Raises
        ------
        NotImplementedError:
            If the worker is not supported
        """
        raise NotImplementedError(f"Please implement the `worker` method")


class TrainerR(Trainer):
    """
    Trainer based on (R)ecorder.
    It will train a list of tasks and return a list of model recorders in a linear way.

    Assumption: models were defined by `task` and the results will be saved to `Recorder`.
    """

    # Those tag will help you distinguish whether the Recorder has finished traning
    STATUS_KEY = "train_status"
    STATUS_BEGIN = "begin_task_train"
    STATUS_END = "end_task_train"

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        train_func: Callable = task_train,
        call_in_subproc: bool = False,
        default_rec_name: Optional[str] = None,
    ):
        """
        Init TrainerR.

        Args:
            experiment_name (str, optional): the default name of experiment.
            train_func (Callable, optional): default training method. Defaults to `task_train`.
            call_in_subproc (bool): call the process in subprocess to force memory release
        """
        super().__init__()
        self.experiment_name = experiment_name
        self.default_rec_name = default_rec_name
        self.train_func = train_func
        self._call_in_subproc = call_in_subproc

    def train(self, tasks: list, train_func: Callable = None, experiment_name: str = None, **kwargs) -> List[Recorder]:
        """
        Given a list of `tasks` and return a list of trained Recorder. The order can be guaranteed.

        Args:
            tasks (list): a list of definitions based on `task` dict
            train_func (Callable): the training method which needs at least `tasks` and `experiment_name`. None for the default training method.
            experiment_name (str): the experiment name, None for use default name.
            kwargs: the params for train_func.

        Returns:
            List[Recorder]: a list of Recorders
        """
        if isinstance(tasks, dict):
            tasks = [tasks]
        if len(tasks) == 0:
            return []
        if train_func is None:
            train_func = self.train_func
        if experiment_name is None:
            experiment_name = self.experiment_name
        recs = []
        for task in tqdm(tasks, desc="train tasks"):
            if self._call_in_subproc:
                get_module_logger("TrainerR").info("running models in sub process (for forcing release memroy).")
                train_func = call_in_subproc(train_func, C)
            rec = train_func(task, experiment_name, recorder_name=self.default_rec_name, **kwargs)
            rec.set_tags(**{self.STATUS_KEY: self.STATUS_BEGIN})
            recs.append(rec)
        return recs

    def end_train(self, models: list, **kwargs) -> List[Recorder]:
        """
        Set STATUS_END tag to the recorders.

        Args:
            models (list): a list of trained recorders.

        Returns:
            List[Recorder]: the same list as the param.
        """
        if isinstance(models, Recorder):
            models = [models]
        for rec in models:
            rec.set_tags(**{self.STATUS_KEY: self.STATUS_END})
        return models


class DelayTrainerR(TrainerR):
    """
    A delayed implementation based on TrainerR, which means `train` method may only do some preparation and `end_train` method can do the real model fitting.
    """

    def __init__(
        self, experiment_name: str = None, train_func=begin_task_train, end_train_func=end_task_train, **kwargs
    ):
        """
        Init TrainerRM.

        Args:
            experiment_name (str): the default name of experiment.
            train_func (Callable, optional): default train method. Defaults to `begin_task_train`.
            end_train_func (Callable, optional): default end_train method. Defaults to `end_task_train`.
        """
        super().__init__(experiment_name, train_func, **kwargs)
        self.end_train_func = end_train_func
        self.delay = True

    def end_train(self, models, end_train_func=None, experiment_name: str = None, **kwargs) -> List[Recorder]:
        """
        Given a list of Recorder and return a list of trained Recorder.
        This class will finish real data loading and model fitting.

        Args:
            models (list): a list of Recorder, the tasks have been saved to them
            end_train_func (Callable, optional): the end_train method which needs at least `recorders` and `experiment_name`. Defaults to None for using self.end_train_func.
            experiment_name (str): the experiment name, None for use default name.
            kwargs: the params for end_train_func.

        Returns:
            List[Recorder]: a list of Recorders
        """
        if isinstance(models, Recorder):
            models = [models]
        if end_train_func is None:
            end_train_func = self.end_train_func
        if experiment_name is None:
            experiment_name = self.experiment_name
        for rec in models:
            if rec.list_tags()[self.STATUS_KEY] == self.STATUS_END:
                continue
            end_train_func(rec, experiment_name, **kwargs)
            rec.set_tags(**{self.STATUS_KEY: self.STATUS_END})
        return models


class TrainerRM(Trainer):
    """
    Trainer based on (R)ecorder and Task(M)anager.
    It can train a list of tasks and return a list of model recorders in a multiprocessing way.

    Assumption: `task` will be saved to TaskManager and `task` will be fetched and trained from TaskManager
    """

    # Those tag will help you distinguish whether the Recorder has finished traning
    STATUS_KEY = "train_status"
    STATUS_BEGIN = "begin_task_train"
    STATUS_END = "end_task_train"

    # This tag is the _id in TaskManager to distinguish tasks.
    TM_ID = "_id in TaskManager"

    def __init__(
        self,
        experiment_name: str = None,
        task_pool: str = None,
        train_func=task_train,
        skip_run_task: bool = False,
        default_rec_name: Optional[str] = None,
    ):
        """
        Init TrainerR.

        Args:
            experiment_name (str): the default name of experiment.
            task_pool (str): task pool name in TaskManager. None for use same name as experiment_name.
            train_func (Callable, optional): default training method. Defaults to `task_train`.
            skip_run_task (bool):
                If skip_run_task == True:
                Only run_task in the worker. Otherwise skip run_task.
        """

        super().__init__()
        self.experiment_name = experiment_name
        self.task_pool = task_pool
        self.train_func = train_func
        self.skip_run_task = skip_run_task
        self.default_rec_name = default_rec_name

    def train(
        self,
        tasks: list,
        train_func: Callable = None,
        experiment_name: str = None,
        before_status: str = TaskManager.STATUS_WAITING,
        after_status: str = TaskManager.STATUS_DONE,
        default_rec_name: Optional[str] = None,
        **kwargs,
    ) -> List[Recorder]:
        """
        Given a list of `tasks` and return a list of trained Recorder. The order can be guaranteed.

        This method defaults to a single process, but TaskManager offered a great way to parallel training.
        Users can customize their train_func to realize multiple processes or even multiple machines.

        Args:
            tasks (list): a list of definitions based on `task` dict
            train_func (Callable): the training method which needs at least `tasks` and `experiment_name`. None for the default training method.
            experiment_name (str): the experiment name, None for use default name.
            before_status (str): the tasks in before_status will be fetched and trained. Can be STATUS_WAITING, STATUS_PART_DONE.
            after_status (str): the tasks after trained will become after_status. Can be STATUS_WAITING, STATUS_PART_DONE.
            kwargs: the params for train_func.

        Returns:
            List[Recorder]: a list of Recorders
        """
        if isinstance(tasks, dict):
            tasks = [tasks]
        if len(tasks) == 0:
            return []
        if train_func is None:
            train_func = self.train_func
        if experiment_name is None:
            experiment_name = self.experiment_name
        if default_rec_name is None:
            default_rec_name = self.default_rec_name
        task_pool = self.task_pool
        if task_pool is None:
            task_pool = experiment_name
        tm = TaskManager(task_pool=task_pool)
        _id_list = tm.create_task(tasks)  # all tasks will be saved to MongoDB
        query = {"_id": {"$in": _id_list}}
        if not self.skip_run_task:
            run_task(
                train_func,
                task_pool,
                query=query,  # only train these tasks
                experiment_name=experiment_name,
                before_status=before_status,
                after_status=after_status,
                recorder_name=default_rec_name,
                **kwargs,
            )

        if not self.is_delay():
            tm.wait(query=query)

        recs = []
        for _id in _id_list:
            rec = tm.re_query(_id)["res"]
            rec.set_tags(**{self.STATUS_KEY: self.STATUS_BEGIN})
            rec.set_tags(**{self.TM_ID: _id})
            recs.append(rec)
        return recs

    def end_train(self, recs: list, **kwargs) -> List[Recorder]:
        """
        Set STATUS_END tag to the recorders.

        Args:
            recs (list): a list of trained recorders.

        Returns:
            List[Recorder]: the same list as the param.
        """
        if isinstance(recs, Recorder):
            recs = [recs]
        for rec in recs:
            rec.set_tags(**{self.STATUS_KEY: self.STATUS_END})
        return recs

    def worker(
        self,
        train_func: Callable = None,
        experiment_name: str = None,
    ):
        """
        The multiprocessing method for `train`. It can share a same task_pool with `train` and can run in other progress or other machines.

        Args:
            train_func (Callable): the training method which needs at least `tasks` and `experiment_name`. None for the default training method.
            experiment_name (str): the experiment name, None for use default name.
        """
        if train_func is None:
            train_func = self.train_func
        if experiment_name is None:
            experiment_name = self.experiment_name
        task_pool = self.task_pool
        if task_pool is None:
            task_pool = experiment_name
        run_task(train_func, task_pool=task_pool, experiment_name=experiment_name)

    def has_worker(self) -> bool:
        return True


class DelayTrainerRM(TrainerRM):
    """
    A delayed implementation based on TrainerRM, which means `train` method may only do some preparation and `end_train` method can do the real model fitting.

    """

    def __init__(
        self,
        experiment_name: str = None,
        task_pool: str = None,
        train_func=begin_task_train,
        end_train_func=end_task_train,
        skip_run_task: bool = False,
        **kwargs,
    ):
        """
        Init DelayTrainerRM.

        Args:
            experiment_name (str): the default name of experiment.
            task_pool (str): task pool name in TaskManager. None for use same name as experiment_name.
            train_func (Callable, optional): default train method. Defaults to `begin_task_train`.
            end_train_func (Callable, optional): default end_train method. Defaults to `end_task_train`.
            skip_run_task (bool):
                If skip_run_task == True:
                Only run_task in the worker. Otherwise skip run_task.
                E.g. Starting trainer on a CPU VM and then waiting tasks to be finished on GPU VMs.
        """
        super().__init__(experiment_name, task_pool, train_func, **kwargs)
        self.end_train_func = end_train_func
        self.delay = True
        self.skip_run_task = skip_run_task

    def train(self, tasks: list, train_func=None, experiment_name: str = None, **kwargs) -> List[Recorder]:
        """
        Same as `train` of TrainerRM, after_status will be STATUS_PART_DONE.

        Args:
            tasks (list): a list of definition based on `task` dict
            train_func (Callable): the train method which need at least `tasks` and `experiment_name`. Defaults to None for using self.train_func.
            experiment_name (str): the experiment name, None for use default name.

        Returns:
            List[Recorder]: a list of Recorders
        """
        if isinstance(tasks, dict):
            tasks = [tasks]
        if len(tasks) == 0:
            return []
        _skip_run_task = self.skip_run_task
        self.skip_run_task = False  # The task preparation can't be skipped
        res = super().train(
            tasks,
            train_func=train_func,
            experiment_name=experiment_name,
            after_status=TaskManager.STATUS_PART_DONE,
            **kwargs,
        )
        self.skip_run_task = _skip_run_task
        return res

    def end_train(self, recs, end_train_func=None, experiment_name: str = None, **kwargs) -> List[Recorder]:
        """
        Given a list of Recorder and return a list of trained Recorder.
        This class will finish real data loading and model fitting.

        Args:
            recs (list): a list of Recorder, the tasks have been saved to them.
            end_train_func (Callable, optional): the end_train method which need at least `recorders` and `experiment_name`. Defaults to None for using self.end_train_func.
            experiment_name (str): the experiment name, None for use default name.
            kwargs: the params for end_train_func.

        Returns:
            List[Recorder]: a list of Recorders
        """
        if isinstance(recs, Recorder):
            recs = [recs]
        if end_train_func is None:
            end_train_func = self.end_train_func
        if experiment_name is None:
            experiment_name = self.experiment_name
        task_pool = self.task_pool
        if task_pool is None:
            task_pool = experiment_name
        _id_list = []
        for rec in recs:
            _id_list.append(rec.list_tags()[self.TM_ID])

        query = {"_id": {"$in": _id_list}}
        if not self.skip_run_task:
            run_task(
                end_train_func,
                task_pool,
                query=query,  # only train these tasks
                experiment_name=experiment_name,
                before_status=TaskManager.STATUS_PART_DONE,
                **kwargs,
            )

        TaskManager(task_pool=task_pool).wait(query=query)

        for rec in recs:
            rec.set_tags(**{self.STATUS_KEY: self.STATUS_END})
        return recs

    def worker(self, end_train_func=None, experiment_name: str = None):
        """
        The multiprocessing method for `end_train`. It can share a same task_pool with `end_train` and can run in other progress or other machines.

        Args:
            end_train_func (Callable, optional): the end_train method which need at least `recorders` and `experiment_name`. Defaults to None for using self.end_train_func.
            experiment_name (str): the experiment name, None for use default name.
        """
        if end_train_func is None:
            end_train_func = self.end_train_func
        if experiment_name is None:
            experiment_name = self.experiment_name
        task_pool = self.task_pool
        if task_pool is None:
            task_pool = experiment_name
        run_task(
            end_train_func,
            task_pool=task_pool,
            experiment_name=experiment_name,
            before_status=TaskManager.STATUS_PART_DONE,
        )

    def has_worker(self) -> bool:
        return True

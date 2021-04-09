# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from qlib.utils import init_instance_by_config, flatten_dict, get_cls_kwargs
from qlib.workflow import R
from qlib.workflow.recorder import Recorder
from qlib.workflow.record_temp import SignalRecord
from qlib.workflow.task.manage import TaskManager, run_task
from qlib.data.dataset import Dataset
from qlib.model.base import Model


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

    # model initiaiton
    model: Model = init_instance_by_config(task_config["model"])
    dataset: Dataset = init_instance_by_config(task_config["dataset"])

    # start exp
    with R.start(experiment_name=experiment_name):

        # train model
        R.log_params(**flatten_dict(task_config))
        model.fit(dataset)
        recorder = R.get_recorder()
        R.save_objects(**{"params.pkl": model})
        R.save_objects(**{"task": task_config})  # keep the original format and datatype
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
                rconf = {"model": model, "dataset": dataset, "recorder": recorder}
            else:
                rconf = {"recorder": recorder}
            r = cls(**kwargs, **rconf)
            r.generate()
    return recorder


class Trainer:
    """
    The trainer which can train a list of model
    """

    def train(self, *args, **kwargs):
        """Given a list of model definition, finished training and return the results of them.

        Returns:
            list: a list of trained results
        """
        raise NotImplementedError(f"Please implement the `train` method.")


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

    def train(self, tasks: list, train_func=None, *args, **kwargs):
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
        run_task(train_func, self.task_pool, experiment_name=self.experiment_name, *args, **kwargs)

        recs = []
        for _id in _id_list:
            recs.append(tm.re_query(_id)["res"])
        return recs

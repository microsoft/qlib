from typing import Dict, Union, List
from qlib import get_module_logger
from qlib.workflow import R
from qlib.model.trainer import task_train
from qlib.workflow.recorder import MLflowRecorder, Recorder
from qlib.workflow.online.update import ModelUpdater
from qlib.workflow.task.utils import TimeAdjuster
from qlib.workflow.task.gen import RollingGen, task_generator
from qlib.workflow.task.manage import TaskManager
from qlib.workflow.task.manage import run_task
from qlib.workflow.task.utils import list_recorders
from qlib.utils.serial import Serializable
from qlib.model.trainer import Trainer, TrainerR


class OnlineManager(Serializable):

    ONLINE_KEY = "online_status"  # the online status key in recorder
    ONLINE_TAG = "online"  # the 'online' model
    NEXT_ONLINE_TAG = "next_online"  # the 'next online' model, which can be 'online' model when call reset_online_model
    OFFLINE_TAG = "offline"  # the 'offline' model, not for online serving

    def __init__(self, trainer: Trainer = None):
        self._trainer = trainer
        self.logger = get_module_logger(self.__class__.__name__)

    def prepare_signals(self, *args, **kwargs):
        raise NotImplementedError(f"Please implement the `prepare_signals` method.")

    def prepare_tasks(self, *args, **kwargs):
        """return the new tasks waiting for training."""
        raise NotImplementedError(f"Please implement the `prepare_tasks` method.")

    def prepare_new_models(self, tasks, *args, **kwargs):
        """Use trainer to train a list of tasks and set the trained model to next_online.

        Args:
            tasks (list): a list of tasks.
        """
        if not (tasks is None or len(tasks) == 0):
            if self._trainer is not None:
                new_models = self._trainer.train(tasks, *args, **kwargs)
                self.set_online_tag(self.NEXT_ONLINE_TAG, new_models)
                self.logger.info(
                    f"Finished prepare {len(new_models)} new models and set them to `{self.NEXT_ONLINE_TAG}`."
                )
            else:
                self.logger.warn("No trainer to train new tasks.")

    def update_online_pred(self, *args, **kwargs):
        raise NotImplementedError(f"Please implement the `update_online_pred` method.")

    def set_online_tag(self, tag, *args, **kwargs):
        """set `tag` to the model to sign whether online

        Args:
            tag (str): the tags in ONLINE_TAG, NEXT_ONLINE_TAG, OFFLINE_TAG
        """
        raise NotImplementedError(f"Please implement the `set_online_tag` method.")

    def get_online_tag(self, *args, **kwargs):
        """given a model and return its online tag"""
        raise NotImplementedError(f"Please implement the `get_online_tag` method.")

    def reset_online_tag(self, *args, **kwargs):
        """offline all models and set the recorders to 'online'. If no parameter and no 'next online' model, then do nothing."""
        raise NotImplementedError(f"Please implement the `reset_online_tag` method.")

    def routine(self, *args, **kwargs):
        """The typical update process in a routine such as day by day or month by month"""
        self.prepare_signals(*args, **kwargs)
        tasks = self.prepare_tasks(*args, **kwargs)
        self.prepare_new_models(tasks, *args, **kwargs)
        self.update_online_pred(*args, **kwargs)
        self.reset_online_tag(*args, **kwargs)


class OnlineManagerR(OnlineManager):
    """
    The implementation of OnlineManager based on (R)ecorder.

    """

    def __init__(self, experiment_name: str, trainer: Trainer = None):
        trainer = TrainerR(experiment_name)
        super().__init__(trainer)
        self.logger = get_module_logger(self.__class__.__name__)
        self.exp_name = experiment_name

    def set_online_tag(self, tag, recorder: Union[Recorder, List]):
        if isinstance(recorder, Recorder):
            recorder = [recorder]
        for rec in recorder:
            rec.set_tags(**{self.ONLINE_KEY: tag})
        self.logger.info(f"Set {len(recorder)} models to '{tag}'.")

    def get_online_tag(self, recorder: Recorder):
        tags = recorder.list_tags()
        return tags.get(OnlineManager.ONLINE_KEY, OnlineManager.OFFLINE_TAG)

    def reset_online_tag(self, recorder: Union[Recorder, List] = None):
        """offline all models and set the recorders to 'online'. If no parameter and no 'next online' model, then do nothing.

        Args:
            recorders (Union[List, Dict], optional):
                the recorders you want to reset to 'online'. If don't give, set 'next online' model to 'online' model. If there isn't any 'next online' model, then maintain existing 'online' model.
        """
        if recorder is None:
            recorder = list(
                list_recorders(
                    self.exp_name, lambda rec: self.get_online_tag(rec) == OnlineManager.NEXT_ONLINE_TAG
                ).values()
            )
        if isinstance(recorder, Recorder):
            recorder = [recorder]
        if len(recorder) == 0:
            self.logger.info("No 'next online' model, just use current 'online' models.")
            return
        recs = list_recorders(self.exp_name)
        self.set_online_tag(OnlineManager.OFFLINE_TAG, list(recs.values()))
        self.set_online_tag(OnlineManager.ONLINE_TAG, recorder)
        self.logger.info(f"Reset {len(recorder)} models to 'online'.")

    def update_online_pred(self, *args, **kwargs):
        """update all online model predictions to the latest day in Calendar"""
        mu = ModelUpdater(self.exp_name)
        cnt = mu.update_all_pred(lambda rec: self.get_online_tag(rec) == OnlineManager.ONLINE_TAG)
        self.logger.info(f"Finish updating {cnt} online model predictions of {self.exp_name}.")


class RollingOnlineManager(OnlineManagerR):
    """An implementation of OnlineManager based on Rolling."""

    def __init__(
        self,
        experiment_name: str,
        rolling_gen: RollingGen,
        trainer: Trainer = None,
    ):
        trainer = TrainerR(experiment_name)
        super().__init__(experiment_name, trainer)
        self.ta = TimeAdjuster()
        self.rg = rolling_gen
        self.logger = get_module_logger(self.__class__.__name__)

    def prepare_signals(self, *args, **kwargs):
        pass

    def prepare_tasks(self, *args, **kwargs):
        """prepare new tasks based on new date.

        Returns:
            list: a list of new tasks.
        """
        latest_records, max_test = self.list_latest_recorders(
            lambda rec: self.get_online_tag(rec) == OnlineManager.ONLINE_TAG
        )
        if max_test is None:
            self.logger.warn(f"No latest online recorders, no new tasks.")
            return []
        calendar_latest = self.ta.last_date()
        if self.ta.cal_interval(calendar_latest, max_test[0]) > self.rg.step:
            old_tasks = []
            for rid, rec in latest_records.items():
                task = rec.load_object("task")
                test_begin = task["dataset"]["kwargs"]["segments"]["test"][0]
                # modify the test segment to generate new tasks
                task["dataset"]["kwargs"]["segments"]["test"] = (test_begin, calendar_latest)
                old_tasks.append(task)
            new_tasks_tmp = task_generator(old_tasks, self.rg)
            new_tasks = [task for task in new_tasks_tmp if task not in old_tasks]
            return new_tasks
        return []

    def list_latest_recorders(self, rec_filter_func=None):
        """find latest recorders based on test segments.

        Args:
            rec_filter_func (Callable, optional): recorder filter. Defaults to None.

        Returns:
            dict, tuple: the latest recorders and the latest date of them
        """
        recs_flt = list_recorders(self.exp_name, rec_filter_func)
        if len(recs_flt) == 0:
            return recs_flt, None
        max_test = max(rec.load_object("task")["dataset"]["kwargs"]["segments"]["test"] for rec in recs_flt.values())
        latest_rec = {}
        for rid, rec in recs_flt.items():
            if rec.load_object("task")["dataset"]["kwargs"]["segments"]["test"] == max_test:
                latest_rec[rid] = rec
        return latest_rec, max_test

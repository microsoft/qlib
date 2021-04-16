from typing import Dict, Union, List
from qlib import get_module_logger
from qlib.workflow import R
from qlib.model.trainer import task_train
from qlib.workflow.recorder import MLflowRecorder, Recorder
from qlib.workflow.online.update import PredUpdater, RecordUpdater
from qlib.workflow.task.collect import Collector
from qlib.workflow.task.utils import TimeAdjuster
from qlib.workflow.task.gen import RollingGen, task_generator
from qlib.workflow.task.manage import TaskManager
from qlib.workflow.task.manage import run_task
from qlib.workflow.task.utils import list_recorders
from qlib.utils.serial import Serializable
from qlib.model.trainer import Trainer, TrainerR
from copy import deepcopy


class OnlineManager:

    ONLINE_KEY = "online_status"  # the online status key in recorder
    ONLINE_TAG = "online"  # the 'online' model
    # NOTE: The meaning of this tag is that we can not assume the training models can be trained before we need its predition. Whenever finished training, it can be guaranteed that there are some online models.
    NEXT_ONLINE_TAG = "next_online"  # the 'next online' model, which can be 'online' model when call reset_online_model
    OFFLINE_TAG = "offline"  # the 'offline' model, not for online serving

    def __init__(self, trainer: Trainer = None, collector: Collector = None, need_log=True):
        """
        init OnlineManager.

        Args:
            trainer (Trainer, optional): a instance of Trainer. Defaults to None.
            collector (Collector, optional): a instance of Collector. Defaults to None.
            need_log (bool, optional): print log or not. Defaults to True.
        """
        self.trainer = trainer
        self.logger = get_module_logger(self.__class__.__name__)
        self.need_log = need_log
        self.delay_signals = {}
        self.collector = collector
        self.cur_time = None

    def prepare_signals(self, *args, **kwargs):
        """
        After perparing the data of last routine (a box in box-plot) which means the end of the routine, we can prepare trading signals for next routine.
        """
        raise NotImplementedError(f"Please implement the `prepare_signals` method.")

    def prepare_tasks(self, *args, **kwargs):
        """
        After the end of a routine, check whether we need to prepare and train some new tasks.
        return the new tasks waiting for training.
        """
        raise NotImplementedError(f"Please implement the `prepare_tasks` method.")

    def prepare_new_models(self, tasks, tag=NEXT_ONLINE_TAG):
        """
        Use trainer to train a list of tasks and set the trained model to `tag`.

        Args:
            tasks (list): a list of tasks.
            tag (str):
                `ONLINE_TAG` for first train or additional train
                `NEXT_ONLINE_TAG` for reset online model when calling `reset_online_tag`
                `OFFLINE_TAG` for train but offline those models
        """
        if not (tasks is None or len(tasks) == 0):
            if self.trainer is not None:
                new_models = self.trainer.train(tasks)
                self.set_online_tag(tag, new_models)
                if self.need_log:
                    self.logger.info(f"Finished prepare {len(new_models)} new models and set them to {tag}.")
            else:
                self.logger.warn("No trainer to train new tasks.")

    def update_online_pred(self, *args, **kwargs):
        """
        After the end of a routine, update the predictions of online models to latest.
        """
        raise NotImplementedError(f"Please implement the `update_online_pred` method.")

    def set_online_tag(self, tag, *args, **kwargs):
        """
        Set `tag` to the model to sign whether online.

        Args:
            tag (str): the tags in `ONLINE_TAG`, `NEXT_ONLINE_TAG`, `OFFLINE_TAG`
        """
        raise NotImplementedError(f"Please implement the `set_online_tag` method.")

    def get_online_tag(self, *args, **kwargs):
        """
        Given a model and return its online tag.
        """
        raise NotImplementedError(f"Please implement the `get_online_tag` method.")

    def reset_online_tag(self, *args, **kwargs):
        """
        Offline all models and set the models to 'online'.
        """
        raise NotImplementedError(f"Please implement the `reset_online_tag` method.")

    def online_models(self):
        """
        Return online models.
        """
        raise NotImplementedError(f"Please implement the `online_models` method.")

    def get_collector(self):
        """
        Return the collector.

        Returns:
            Collector
        """
        return self.collector

    def run_delay_signals(self):
        """
        Prepare all signals if there are some dates waiting for prepare.
        """
        for cur_time, params in self.delay_signals.items():
            self.cur_time = cur_time
            self.prepare_signals(*params[0], **params[1])
        self.delay_signals = {}

    def routine(self, cur_time=None, delay_prepare=False, *args, **kwargs):
        """
        The typical update process after a routine, such as day by day or month by month.
        Prepare signals -> prepare tasks -> prepare new models -> update online prediction -> reset online models
        """
        self.cur_time = cur_time  # None for latest date
        if not delay_prepare:
            self.prepare_signals(*args, **kwargs)
        else:
            if cur_time is not None:
                self.delay_signals[cur_time] = (args, kwargs)
            else:
                raise ValueError("Can not delay prepare when cur_time is None")
        tasks = self.prepare_tasks(*args, **kwargs)
        self.prepare_new_models(tasks)
        self.update_online_pred()
        return self.reset_online_tag()


class OnlineManagerR(OnlineManager):
    """
    The implementation of OnlineManager based on (R)ecorder.

    """

    def __init__(self, experiment_name: str, trainer: Trainer = None, collector: Collector = None, need_log=True):
        """
        init OnlineManagerR.

        Args:
            experiment_name (str): the experiment name.
            trainer (Trainer, optional): a instance of Trainer. Defaults to None.
            collector (Collector, optional): a instance of Collector. Defaults to None.
            need_log (bool, optional): print log or not. Defaults to True.
        """
        trainer = TrainerR(experiment_name)
        super().__init__(trainer=trainer, collector=collector, need_log=need_log)
        self.exp_name = experiment_name

    def set_online_tag(self, tag, recorder: Union[Recorder, List]):
        if isinstance(recorder, Recorder):
            recorder = [recorder]
        for rec in recorder:
            rec.set_tags(**{self.ONLINE_KEY: tag})
        if self.need_log:
            self.logger.info(f"Set {len(recorder)} models to '{tag}'.")

    def get_online_tag(self, recorder: Recorder):
        tags = recorder.list_tags()
        return tags.get(OnlineManager.ONLINE_KEY, OnlineManager.OFFLINE_TAG)

    def reset_online_tag(self, recorder: Union[Recorder, List] = None):
        """offline all models and set the recorders to 'online'. If no parameter and no 'next online' model, then do nothing.

        Args:
            recorders (Union[List, Dict], optional):
                the recorders you want to reset to 'online'. If don't give, set 'next online' model to 'online' model. If there isn't any 'next online' model, then maintain existing 'online' model.

        Returns:
            list: new online recorder. [] if there is no update.
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
            if self.need_log:
                self.logger.info("No 'next online' model, just use current 'online' models.")
            return []
        recs = list_recorders(self.exp_name)
        self.set_online_tag(OnlineManager.OFFLINE_TAG, list(recs.values()))
        self.set_online_tag(OnlineManager.ONLINE_TAG, recorder)
        return recorder

    def online_models(self):
        return list(
            list_recorders(self.exp_name, lambda rec: self.get_online_tag(rec) == OnlineManager.ONLINE_TAG).values()
        )

    def update_online_pred(self):
        """
        Update all online model predictions to the latest day in Calendar
        """
        online_models = self.online_models()
        for rec in online_models:
            PredUpdater(rec, to_date=self.cur_time, need_log=self.need_log).update()

        if self.need_log:
            self.logger.info(f"Finish updating {len(online_models)} online model predictions of {self.exp_name}.")


class RollingOnlineManager(OnlineManagerR):
    """An implementation of OnlineManager based on Rolling."""

    def __init__(
        self,
        experiment_name: str,
        rolling_gen: RollingGen,
        trainer: Trainer = None,
        collector: Collector = None,
        need_log=True,
    ):
        """
        init RollingOnlineManager.

        Args:
            experiment_name (str): the experiment name.
            rolling_gen (RollingGen): a instance of RollingGen
            trainer (Trainer, optional): a instance of Trainer. Defaults to None.
            collector (Collector, optional): a instance of Collector. Defaults to None.
            need_log (bool, optional): print log or not. Defaults to True.
        """
        trainer = TrainerR(experiment_name)
        super().__init__(experiment_name=experiment_name, trainer=trainer, collector=collector, need_log=need_log)
        self.ta = TimeAdjuster()
        self.rg = rolling_gen
        self.logger = get_module_logger(self.__class__.__name__)

    def prepare_signals(self, *args, **kwargs):
        """
        Must use `pass` even though there is nothing to do.
        """
        pass

    def prepare_tasks(self, *args, **kwargs):
        """
        Prepare new tasks based on new date.

        Returns:
            list: a list of new tasks.
        """
        self.ta.set_end_time(self.cur_time)
        latest_records, max_test = self.list_latest_recorders(
            lambda rec: self.get_online_tag(rec) == OnlineManager.ONLINE_TAG
        )
        if max_test is None:
            self.logger.warn(f"No latest online recorders, no new tasks.")
            return []
        calendar_latest = self.ta.last_date() if self.cur_time is None else self.cur_time
        if self.need_log:
            self.logger.info(
                f"The interval between current time and last rolling test begin time is {self.ta.cal_interval(calendar_latest, max_test[0])}, the rolling step is {self.rg.step}"
            )
        if self.ta.cal_interval(calendar_latest, max_test[0]) >= self.rg.step:
            old_tasks = []
            tasks_tmp = []
            for rid, rec in latest_records.items():
                task = rec.load_object("task")
                old_tasks.append(deepcopy(task))
                test_begin = task["dataset"]["kwargs"]["segments"]["test"][0]
                # modify the test segment to generate new tasks
                task["dataset"]["kwargs"]["segments"]["test"] = (test_begin, calendar_latest)
                tasks_tmp.append(task)
            new_tasks_tmp = task_generator(tasks_tmp, self.rg)
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

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This class is a component of online serving, it can manage a series of models dynamically.
With the change of time, the decisive models will be also changed. In this module, we called those contributing models as `online` models.
In every routine(such as everyday or every minutes), the `online` models maybe changed and the prediction of them need to be updated.
So this module provide a series methods to control this process. 
"""
from copy import deepcopy
from pprint import pprint
import pandas as pd
from qlib.model.ens.ensemble import ens_workflow
from qlib.model.ens.group import RollingGroup
from qlib.utils.serial import Serializable
from typing import Dict, List, Union
from qlib import get_module_logger
from qlib.data.data import D
from qlib.model.trainer import Trainer, TrainerR, task_train
from qlib.workflow import R
from qlib.workflow.online.strategy import OnlineStrategy
from qlib.workflow.online.update import PredUpdater
from qlib.workflow.recorder import Recorder
from qlib.workflow.task.collect import Collector, HyperCollector, RecorderCollector
from qlib.workflow.task.gen import RollingGen, task_generator
from qlib.workflow.task.utils import TimeAdjuster, list_recorders

class OnlineManager(Serializable):

    ONLINE_KEY = "online_status"  # the online status key in recorder
    ONLINE_TAG = "online"  # the 'online' model
    # NOTE: The meaning of this tag is that we can not assume the training models can be trained before we need its predition. Whenever finished training, it can be guaranteed that there are some online models.
    NEXT_ONLINE_TAG = "next_online"  # the 'next online' model, which can be 'online' model when call reset_online_model
    OFFLINE_TAG = "offline"  # the 'offline' model, not for online serving

    SIGNAL_EXP = "OnlineManagerSignals"  # a specific experiment to save signals of different experiment.

    def __init__(self, trainer: Trainer = None, need_log=True):
        """
        init OnlineManager.

        Args:
            trainer (Trainer, optional): a instance of Trainer. Defaults to None.
            need_log (bool, optional): print log or not. Defaults to True.
        """
        self.trainer = trainer
        self.logger = get_module_logger(self.__class__.__name__)
        self.need_log = need_log
        self.cur_time = None

    def prepare_signals(self):
        """
        After perparing the data of last routine (a box in box-plot) which means the end of the routine, we can prepare trading signals for next routine.
        Must use `pass` even though there is nothing to do.
        """
        raise NotImplementedError(f"Please implement the `prepare_signals` method.")

    def get_signals(self):
        """
        After preparing signals, here is the method to get them.
        """
        raise NotImplementedError(f"Please implement the `get_signals` method.")

    def prepare_tasks(self, *args, **kwargs):
        """
        After the end of a routine, check whether we need to prepare and train some new tasks.
        return the new tasks waiting for training.
        """
        raise NotImplementedError(f"Please implement the `prepare_tasks` method.")

    def prepare_new_models(self, tasks, tag=NEXT_ONLINE_TAG, check_func=None, *args, **kwargs):
        """
        Use trainer to train a list of tasks and set the trained model to `tag`.

        Args:
            tasks (list): a list of tasks.
            tag (str):
                `ONLINE_TAG` for first train or additional train
                `NEXT_ONLINE_TAG` for reset online model when calling `reset_online_tag`
                `OFFLINE_TAG` for train but offline those models
            check_func: the method to judge if a model can be online.
                The parameter is the model record and return True for online.
                None for online every models.
            *args, **kwargs: will be passed to end_train which means will be passed to customized train method.

        """
        if check_func is None:
            check_func = lambda x: True
        if len(tasks) > 0:
            if self.trainer is not None:
                new_models = self.trainer.train(tasks, *args, **kwargs)
                if check_func(new_models):
                    self.set_online_tag(tag, new_models)
                    if self.need_log:
                        self.logger.info(f"Finished preparing {len(new_models)} new models and set them to {tag}.")
            else:
                self.logger.warn("No trainer to train new tasks.")

    def update_online_pred(self):
        """
        After the end of a routine, update the predictions of online models to latest.
        """
        raise NotImplementedError(f"Please implement the `update_online_pred` method.")

    def set_online_tag(self, tag, recorder):
        """
        Set `tag` to the model to sign whether online.

        Args:
            tag (str): the tags in `ONLINE_TAG`, `NEXT_ONLINE_TAG`, `OFFLINE_TAG`
        """
        raise NotImplementedError(f"Please implement the `set_online_tag` method.")

    def get_online_tag(self):
        """
        Given a model and return its online tag.
        """
        raise NotImplementedError(f"Please implement the `get_online_tag` method.")

    def reset_online_tag(self, recorders=None):
        """offline all models and set the recorders to 'online'. If no parameter and no 'next online' model, then do nothing.

        Args:
            recorders (List, optional):
                the recorders you want to reset to 'online'. If don't give, set 'next online' model to 'online' model. If there isn't any 'next online' model, then maintain existing 'online' model.

        Returns:
            list: new online recorder. [] if there is no update.
        """
        raise NotImplementedError(f"Please implement the `reset_online_tag` method.")

    def online_models(self):
        """
        Return online models.
        """
        raise NotImplementedError(f"Please implement the `online_models` method.")

    def first_train(self):
        """
        Train a series of models firstly and set some of them into online models.
        """
        raise NotImplementedError(f"Please implement the `first_train` method.")

    def get_collector(self):
        """
        Return the collector.

        Returns:
            Collector
        """
        raise NotImplementedError(f"Please implement the `get_collector` method.")

    def delay_prepare(self, rec_dict, *args, **kwargs):
        """
        Prepare all models and signals if there are something waiting for prepare.
        NOTE: Assumption: the predictions of online models are between `time_segment`, or this method will work in a wrong way.

        Args:
            rec_dict (str): an online models dict likes {(begin_time, end_time):[online models]}.
            *args, **kwargs: will be passed to end_train which means will be passed to customized train method.
        """
        for time_segment, recs_list in rec_dict.items():
            self.trainer.end_train(recs_list, *args, **kwargs)
            self.reset_online_tag(recs_list)
            self.prepare_signals()
            signal_max = self.get_signals().index.get_level_values("datetime").max()
            if time_segment[1] is not None and signal_max > time_segment[1]:
                raise ValueError(
                    f"The max time of signals prepared by online models is {signal_max}, but those models only online in {time_segment}"
                )

    def routine(self, cur_time=None, delay_prepare=False, *args, **kwargs):
        """
        The typical update process after a routine, such as day by day or month by month.
        update online prediction -> prepare signals -> prepare tasks -> prepare new models -> reset online models

        NOTE: Assumption: if using simulator (delay_prepare is True), the prediction will be prepared well after every training, so there is no need to update predictions.

        Args:
            cur_time ([type], optional): [description]. Defaults to None.
            delay_prepare (bool, optional): [description]. Defaults to False.
            *args, **kwargs: will be passed to `prepare_tasks` and `prepare_new_models`. It can be some hyper parameter or training config.

        Returns:
            [type]: [description]
        """
        self.cur_time = cur_time  # None for latest date
        if not delay_prepare:
            self.update_online_pred()
            self.prepare_signals()
        tasks = self.prepare_tasks(*args, **kwargs)
        self.prepare_new_models(tasks, *args, **kwargs)

        return self.reset_online_tag()


class OnlineManagerR(OnlineManager):
    """
    The implementation of OnlineManager based on (R)ecorder.

    """

    def __init__(self, experiment_name: str, trainer: Trainer = None, need_log=True):
        """
        init OnlineManagerR.

        Args:
            experiment_name (str): the experiment name.
            trainer (Trainer, optional): a instance of Trainer. Defaults to None.
            need_log (bool, optional): print log or not. Defaults to True.
        """
        if trainer is None:
            trainer = TrainerR(experiment_name)
        super().__init__(trainer=trainer, need_log=need_log)
        self.exp_name = experiment_name
        self.signal_rec = None

    def set_online_tag(self, tag, recorder: Union[Recorder, List]):
        """
        Set `tag` to the model to sign whether online.

        Args:
            tag (str): the tags in `ONLINE_TAG`, `NEXT_ONLINE_TAG`, `OFFLINE_TAG`
            recorder (Union[Recorder, List])
        """
        if isinstance(recorder, Recorder):
            recorder = [recorder]
        for rec in recorder:
            rec.set_tags(**{self.ONLINE_KEY: tag})
        if self.need_log:
            self.logger.info(f"Set {len(recorder)} models to '{tag}'.")

    def get_online_tag(self, recorder: Recorder):
        """
        Given a model and return its online tag.

        Args:
            recorder (Recorder): a instance of recorder

        Returns:
            str: the tag
        """
        tags = recorder.list_tags()
        return tags.get(OnlineManager.ONLINE_KEY, OnlineManager.OFFLINE_TAG)

    def reset_online_tag(self, recorder: Union[Recorder, List] = None):
        """offline all models and set the recorders to 'online'. If no parameter and no 'next online' model, then do nothing.

        Args:
            recorders (Union[Recorder, List], optional):
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

    def get_signals(self):
        """
        get signals from the recorder(named self.exp_name) of the experiment(named self.SIGNAL_EXP)

        Returns:
            signals
        """
        if self.signal_rec is None:
            with R.start(experiment_name=self.SIGNAL_EXP, recorder_name=self.exp_name, resume=True):
                self.signal_rec = R.get_recorder()
        signals = None
        try:
            signals = self.signal_rec.load_object("signals")
        except OSError:
            self.logger.warn("Can not find `signals`, have you called `prepare_signals` before?")
        return signals

    def online_models(self):
        """
        Return online models.

        Returns:
            list: the list of online models
        """
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
            self.logger.info(f"Finished updating {len(online_models)} online model predictions of {self.exp_name}.")

    def prepare_signals(self, over_write=False):
        """
        Average the predictions of online models and offer a trading signals every routine.
        The signals will be saved to `signal` file of a recorder named self.exp_name of a experiment using the name of `SIGNAL_EXP`
        Even if the latest signal already exists, the latest calculation result will be overwritten.
        NOTE: Given a prediction of a certain time, all signals before this time will be prepared well.
        Args:
            over_write (bool, optional): If True, the new signals will overwrite the file. If False, the new signals will append to the end of signals. Defaults to False.
        """
        if self.signal_rec is None:
            with R.start(experiment_name=self.SIGNAL_EXP, recorder_name=self.exp_name, resume=True):
                self.signal_rec = R.get_recorder()

        pred = []
        try:
            old_signals = self.signal_rec.load_object("signals")
        except OSError:
            old_signals = None

        for rec in self.online_models():
            pred.append(rec.load_object("pred.pkl"))

        signals = pd.concat(pred, axis=1).mean(axis=1).to_frame("score")
        signals = signals.sort_index()
        if old_signals is not None and not over_write:
            old_max = old_signals.index.get_level_values("datetime").max()
            new_signals = signals.loc[old_max:]
            signals = pd.concat([old_signals, new_signals], axis=0)
        else:
            new_signals = signals
        if self.need_log:
            self.logger.info(f"Finished preparing new {len(new_signals)} signals to {self.SIGNAL_EXP}/{self.exp_name}.")
        self.signal_rec.save_objects(**{"signals": signals})


class RollingOnlineManager(OnlineManagerR):
    """An implementation of OnlineManager based on Rolling."""

    def __init__(
        self,
        experiment_name: str,
        rolling_gen: RollingGen,
        trainer: Trainer = None,
        need_log=True,
    ):
        """
        init RollingOnlineManager.

        Args:
            experiment_name (str): the experiment name.
            rolling_gen (RollingGen): an instance of RollingGen
            trainer (Trainer, optional): an instance of Trainer. Defaults to None.
            collector (Collector, optional): an instance of Collector. Defaults to None.
            need_log (bool, optional): print log or not. Defaults to True.
        """
        if trainer is None:
            trainer = TrainerR(experiment_name)
        super().__init__(experiment_name=experiment_name, trainer=trainer, need_log=need_log)
        self.ta = TimeAdjuster()
        self.rg = rolling_gen
        self.logger = get_module_logger(self.__class__.__name__)

    def get_collector(self, rec_key_func=None, rec_filter_func=None):
        """
        Get the instance of collector to collect results. The returned collector must can distinguish results in different models.
        Assumption: the models can be distinguished based on model name and rolling test segments.
        If you do not want this assumption, please implement your own method or use another rec_key_func.

        Args:
            rec_key_func (Callable): a function to get the key of a recorder. If None, use recorder id.
            rec_filter_func (Callable, optional): filter the recorder by return True or False. Defaults to None.
        """

        def rec_key(recorder):
            task_config = recorder.load_object("task")
            model_key = task_config["model"]["class"]
            rolling_key = task_config["dataset"]["kwargs"]["segments"]["test"]
            return model_key, rolling_key

        if rec_key_func is None:
            rec_key_func = rec_key

        return RecorderCollector(experiment=self.exp_name, rec_key_func=rec_key_func, rec_filter_func=rec_filter_func)

    def collect_artifact(self, rec_key_func=None, rec_filter_func=None):
        """
        collecting artifact based on the collector and RollingGroup.

        Args:
            rec_key_func (Callable): a function to get the key of a recorder. If None, use recorder id.
            rec_filter_func (Callable, optional): filter the recorder by return True or False. Defaults to None.

        Returns:
            dict: the artifact dict after rolling ensemble
        """
        artifact = ens_workflow(
            self.get_collector(rec_key_func=rec_key_func, rec_filter_func=rec_filter_func), RollingGroup()
        )
        return artifact

    def first_train(self, task_configs: list):
        """
        Use rolling_gen to generate different tasks based on task_configs and trained them.

        Args:
            task_configs (list or dict): a list of task configs or a task config

        Returns:
            Collector: a instance of a Collector.
        """
        tasks = task_generator(
            tasks=task_configs,
            generators=self.rg,  # generate different date segment
        )
        self.prepare_new_models(tasks, tag=self.ONLINE_TAG)
        return self.get_collector()

    def prepare_tasks(self):
        """
        Prepare new tasks based on new date.

        Returns:
            list: a list of new tasks.
        """
        latest_records, max_test = self.list_latest_recorders(
            lambda rec: self.get_online_tag(rec) == OnlineManager.ONLINE_TAG
        )
        if max_test is None:
            self.logger.warn(f"No latest online recorders, no new tasks.")
            return []
        calendar_latest = D.calendar(end_time=self.cur_time)[-1] if self.cur_time is None else self.cur_time
        if self.need_log:
            self.logger.info(
                f"The interval between current time {calendar_latest} and last rolling test begin time {max_test[0]} is {self.ta.cal_interval(calendar_latest, max_test[0])}, the rolling step is {self.rg.step}"
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


class OnlineM(Serializable):
    def __init__(
        self, strategy: Union[OnlineStrategy, List[OnlineStrategy]], begin_time=None, freq="day", need_log=True
    ):
        self.logger = get_module_logger(self.__class__.__name__)
        self.need_log = need_log
        if not isinstance(strategy, list):
            strategy = [strategy]
        self.strategy = strategy
        self.freq = freq
        if begin_time is None:
            begin_time = D.calendar(freq=self.freq).max()
        self.cur_time = pd.Timestamp(begin_time)
        self.history = {}

    def first_train(self):
        """
        Train a series of models firstly and set some of them into online models.
        """
        for strategy in self.strategy:
            self.logger.info(f"Strategy `{strategy.name_id}` begins first training...")
            online_models = strategy.first_train()
            self.history.setdefault(strategy.name_id, {})[self.cur_time] = online_models

    def routine(self, cur_time=None, task_kwargs={}, model_kwargs={}):
        """
        The typical update process after a routine, such as day by day or month by month.
        update online prediction -> prepare signals -> prepare tasks -> prepare new models -> reset online models

        NOTE: Assumption: if using simulator (delay_prepare is True), the prediction will be prepared well after every training, so there is no need to update predictions.

        Args:
            cur_time ([type], optional): [description]. Defaults to None.
            delay_prepare (bool, optional): [description]. Defaults to False.
            *args, **kwargs: will be passed to `prepare_tasks` and `prepare_new_models`. It can be some hyper parameter or training config.

        Returns:
            [type]: [description]
        """
        if cur_time is None:
            cur_time = D.calendar(freq=self.freq).max()
        self.cur_time = pd.Timestamp(cur_time)  # None for latest date
        for strategy in self.strategy:
            self.logger.info(f"Strategy `{strategy.name_id}` begins routine...")
            if not strategy.trainer.is_delay():
                strategy.prepare_signals()
            tasks = strategy.prepare_tasks(self.cur_time, **task_kwargs)
            online_models = strategy.prepare_online_models(tasks, **model_kwargs)
            if len(online_models) > 0:
                self.history.setdefault(strategy.name_id, {})[self.cur_time] = online_models

    def get_collector(self):
        collector_dict = {}
        for strategy in self.strategy:
            collector_dict[strategy.name_id] = strategy.get_collector()
        return HyperCollector(collector_dict)

    def get_online_history(self, strategy_name_id):
        history_dict = self.history[strategy_name_id]
        history = []
        for time in sorted(history_dict):
            models = history_dict[time]
            history.append((time, models))
        return history

    def delay_prepare(self, delay_kwargs={}):
        """
        Prepare all models and signals if there are something waiting for prepare.
        NOTE: Assumption: the predictions of online models are between `time_segment`, or this method will work in a wrong way.

        Args:
            rec_dict (str): an online models dict likes {(begin_time, end_time):[online models]}.
            *args, **kwargs: will be passed to end_train which means will be passed to customized train method.
        """
        for strategy in self.strategy:
            strategy.delay_prepare(self.get_online_history(strategy.name_id), **delay_kwargs)

    def simulate(self, end_time, frequency="day", task_kwargs={}, model_kwargs={}, delay_kwargs={}):
        """
        Starting from start time, this method will simulate every routine in OnlineManager.
        NOTE: Considering the parallel training, the models and signals can be perpared after all routine simulating.

        Returns:
            Collector: the OnlineManager's collector
        """
        cal = D.calendar(start_time=self.cur_time, end_time=end_time, freq=frequency)
        self.first_train()
        for cur_time in cal:
            self.logger.info(f"Simulating at {str(cur_time)}......")
            self.routine(cur_time, task_kwargs=task_kwargs, model_kwargs=model_kwargs)
        self.delay_prepare(delay_kwargs=delay_kwargs)
        self.logger.info(f"Finished preparing signals")
        return self.get_collector()

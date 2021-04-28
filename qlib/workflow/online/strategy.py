# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
This module is working with OnlineManager, responsing for a set of strategy about how the models are updated and signals are perpared.
"""

from copy import deepcopy
from typing import List, Union
import pandas as pd
from qlib.data.data import D
from qlib.log import get_module_logger
from qlib.model.ens.group import RollingGroup
from qlib.model.trainer import Trainer, TrainerR
from qlib.workflow import R
from qlib.workflow.online.utils import OnlineTool, OnlineToolR
from qlib.workflow.task.collect import HyperCollector, RecorderCollector
from qlib.workflow.task.gen import RollingGen, task_generator
from qlib.workflow.task.utils import TimeAdjuster, list_recorders


class OnlineStrategy:
    def __init__(self, name_id: str, trainer: Trainer = None, need_log=True):
        """
        init OnlineManager.

        Args:
            name_id (str): a unique name or id
            trainer (Trainer, optional): a instance of Trainer. Defaults to None.
            need_log (bool, optional): print log or not. Defaults to True.
        """
        self.name_id = name_id
        self.trainer = trainer
        self.logger = get_module_logger(self.__class__.__name__)
        self.need_log = need_log
        self.tool = OnlineTool()
        self.history = {}

    def prepare_signals(self, delay=False):
        """
        After perparing the data of last routine (a box in box-plot) which means the end of the routine, we can prepare trading signals for next routine.
        Must use `pass` even though there is nothing to do.
        """
        raise NotImplementedError(f"Please implement the `prepare_signals` method.")

    def prepare_tasks(self, *args, **kwargs):
        """
        After the end of a routine, check whether we need to prepare and train some new tasks.
        return the new tasks waiting for training.
        """
        raise NotImplementedError(f"Please implement the `prepare_tasks` method.")

    def prepare_online_models(self, tasks, check_func=None, **kwargs):
        """
        Use trainer to train a list of tasks and set the trained model to `online`.

        Args:
            tasks (list): a list of tasks.
            tag (str):
                `ONLINE_TAG` for first train or additional train
                `NEXT_ONLINE_TAG` for reset online model when calling `reset_online_tag`
                `OFFLINE_TAG` for train but offline those models
            check_func: the method to judge if a model can be online.
                The parameter is the model record and return True for online.
                None for online every models.
            **kwargs: will be passed to end_train which means will be passed to customized train method.

        """
        if check_func is None:
            check_func = lambda x: True
        online_models = []
        if len(tasks) > 0:
            new_models = self.trainer.train(tasks, **kwargs)
            for model in new_models:
                if check_func(model):
                    online_models.append(model)
            self.tool.reset_online_tag(online_models)
        return online_models

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

    def delay_prepare(self, history, **kwargs):
        """
        Prepare all models and signals if there are something waiting for prepare.
        NOTE: Assumption: the predictions of online models are between `time_segment`, or this method will work in a wrong way.

        Args:
            rec_dict (str): an online models dict likes {(begin_time, end_time):[online models]}.
            *args, **kwargs: will be passed to end_train which means will be passed to customized train method.
        """
        for time_begin, recs_list in history:
            self.trainer.end_train(recs_list, **kwargs)
            self.tool.reset_online_tag(recs_list)
            self.prepare_signals(delay=True)


class RollingAverageStrategy(OnlineStrategy):

    """
    This example strategy always use latest rolling model as online model and prepare trading signals using the average prediction of online models
    """

    def __init__(
        self,
        name_id: str,
        task_template: Union[dict, List[dict]],
        rolling_gen: RollingGen,
        trainer: Trainer = None,
        need_log=True,
        signal_exp_name="OnlineManagerSignals",
    ):
        """
        init OnlineManagerR.

        Assumption: the str of name_id, the experiment name and the trainer's experiment name are same one.

        Args:
            name_id (str): a unique name or id. Will be also the name of Experiment.
            task_template (Union[dict,List[dict]]): a list of task_template or a single template, which will be used to generate many tasks using rolling_gen.
            rolling_gen (RollingGen): an instance of RollingGen
            trainer (Trainer, optional): a instance of Trainer. Defaults to None.
            need_log (bool, optional): print log or not. Defaults to True.
            signal_exp_path (str): a specific experiment to save signals of different experiment.
        """
        super().__init__(name_id=name_id, trainer=trainer, need_log=need_log)
        self.exp_name = self.name_id
        if not isinstance(task_template, list):
            task_template = [task_template]
        self.task_template = task_template
        self.signal_rec = None
        self.signal_exp_name = signal_exp_name
        self.ta = TimeAdjuster()
        self.rg = rolling_gen
        self.tool = OnlineToolR(self.exp_name)

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

        artifacts_collector = RecorderCollector(
            experiment=self.exp_name,
            process_list=RollingGroup(),
            rec_key_func=rec_key_func,
            rec_filter_func=rec_filter_func,
        )

        signals_collector = RecorderCollector(
            experiment=self.signal_exp_name,
            rec_key_func=lambda rec: rec.info["name"],
            rec_filter_func=lambda rec: rec.info["name"] == self.exp_name,
            artifacts_path={"signals": "signals"},
        )
        return HyperCollector({"artifacts": artifacts_collector, "signals": signals_collector})

    def first_train(self):
        """
        Use rolling_gen to generate different tasks based on task_template and trained them.

        Returns:
            Collector: a instance of a Collector.
        """
        tasks = task_generator(
            tasks=self.task_template,
            generators=self.rg,  # generate different date segment
        )
        return self.prepare_online_models(tasks)

    def prepare_tasks(self, cur_time):
        """
        Prepare new tasks based on cur_time (None for latest).

        Returns:
            list: a list of new tasks.
        """
        latest_records, max_test = self._list_latest(self.tool.online_models())
        if max_test is None:
            self.logger.warn(f"No latest online recorders, no new tasks.")
            return []
        calendar_latest = D.calendar(end_time=cur_time)[-1] if cur_time is None else cur_time
        if self.need_log:
            self.logger.info(
                f"The interval between current time {calendar_latest} and last rolling test begin time {max_test[0]} is {self.ta.cal_interval(calendar_latest, max_test[0])}, the rolling step is {self.rg.step}"
            )
        if self.ta.cal_interval(calendar_latest, max_test[0]) >= self.rg.step:
            old_tasks = []
            tasks_tmp = []
            for rec in latest_records:
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

    def prepare_signals(self, delay=False, over_write=False):
        """
        Average the predictions of online models and offer a trading signals every routine.
        The signals will be saved to `signal` file of a recorder named self.exp_name of a experiment using the name of `SIGNAL_EXP`
        Even if the latest signal already exists, the latest calculation result will be overwritten.
        NOTE: Given a prediction of a certain time, all signals before this time will be prepared well.
        Args:
            over_write (bool, optional): If True, the new signals will overwrite the file. If False, the new signals will append to the end of signals. Defaults to False.
        Returns:
            object: the signals.
        """
        if not delay:
            self.tool.update_online_pred()
        if self.signal_rec is None:
            with R.start(experiment_name=self.signal_exp_name, recorder_name=self.exp_name, resume=True):
                self.signal_rec = R.get_recorder()

        pred = []
        try:
            old_signals = self.signal_rec.load_object("signals")
        except OSError:
            old_signals = None

        for rec in self.tool.online_models():
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
            self.logger.info(
                f"Finished preparing new {len(new_signals)} signals to {self.signal_exp_name}/{self.exp_name}."
            )
        self.signal_rec.save_objects(**{"signals": signals})
        return signals

        # def get_signals(self):
        # """
        # get signals from the recorder(named self.exp_name) of the experiment(named self.SIGNAL_EXP)

        # Returns:
        #     signals
        # """
        # if self.signal_rec is None:
        #     with R.start(experiment_name=self.signal_exp_name, recorder_name=self.exp_name, resume=True):
        #         self.signal_rec = R.get_recorder()
        # signals = None
        # try:
        #     signals = self.signal_rec.load_object("signals")
        # except OSError:
        #     self.logger.warn("Can not find `signals`, have you called `prepare_signals` before?")
        # return signals

    def _list_latest(self, rec_list):
        if len(rec_list) == 0:
            return rec_list, None
        max_test = max(rec.load_object("task")["dataset"]["kwargs"]["segments"]["test"] for rec in rec_list)
        latest_rec = []
        for rec in rec_list:
            if rec.load_object("task")["dataset"]["kwargs"]["segments"]["test"] == max_test:
                latest_rec.append(rec)
        return latest_rec, max_test

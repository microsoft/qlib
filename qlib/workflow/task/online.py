from typing import Dict, Union, List
from qlib import get_module_logger
from qlib.workflow import R
from qlib.model.trainer import task_train
from qlib.workflow.recorder import MLflowRecorder, Recorder
from qlib.workflow.task.collect import TaskCollector
from qlib.workflow.task.update import ModelUpdater
from qlib.workflow.task.utils import TimeAdjuster
from qlib.workflow.task.gen import RollingGen, task_generator
from qlib.workflow.task.manage import TaskManager
from qlib.workflow.task.manage import run_task


class OnlineManager:
    def prepare_new_models(self, tasks: List[dict]):
        """prepare(train) new models

        Parameters
        ----------
        tasks : List[dict]
            a list of tasks

        """
        raise NotImplementedError(f"Please implement the `prepare_new_models` method.")

    ONLINE_KEY = "online_status"  # the tag key in recorder
    ONLINE_TAG = "online"  # the 'online' model
    NEXT_ONLINE_TAG = "next_online"  # the 'next online' model, which can be 'online' model when call reset_online_model
    OFFLINE_TAG = "offline"  # the 'offline' model, not for online serving

    def __init__(self, experiment_name: str) -> None:
        """ModelUpdater needs experiment name to find the records

        Parameters
        ----------
        experiment_name : str
            experiment name string
        """
        self.logger = get_module_logger("OnlineManagement")
        self.exp_name = experiment_name
        self.tc = TaskCollector(experiment_name)

    def set_next_online_model(self, recorder: MLflowRecorder):
        recorder.set_tags(**{self.ONLINE_KEY: self.NEXT_ONLINE_TAG})

    def set_online_model(self, recorder: MLflowRecorder):
        """online model will be identified at the tags of the record"""
        recorder.set_tags(**{self.ONLINE_KEY: self.ONLINE_TAG})

    def set_offline_model(self, recorder: MLflowRecorder):
        recorder.set_tags(**{self.ONLINE_KEY: self.OFFLINE_TAG})

    def offline_all_model(self):
        recs = self.tc.list_recorders()
        for rid, rec in recs.items():
            self.set_offline_model(rec)

    def reset_online_model(self, recorders: Union[List, Dict] = None):
        """offline all models and set the recorders to 'online'. If no parameter and no 'next online' model, then do nothing.

        Args:
            recorders (Union[List, Dict], optional):
                the recorders you want to reset to 'online'. If don't give, set 'next online' model to 'online' model. If there isn't any 'next online' model, then maintain existing 'online' model.
        """
        if recorders is None:
            recorders = self.list_next_online_model()
        if len(recorders) == 0:
            self.logger.info("No 'next online' model, just use current 'online' models.")
            return
        self.offline_all_model()
        if isinstance(recorders, dict):
            recorders = recorders.values()
        for rec in recorders:
            self.set_online_model(rec)
        self.logger.info(f"Reset {len(recorders)} models to 'online'.")

    def set_latest_model_to_next_online(self):
        latest_rec = self.tc.list_latest_recorders()
        for rid, rec in latest_rec.items():
            self.set_next_online_model(rec)
        self.logger.info(f"Set {len(latest_rec)} latest models to 'next online'.")

    @staticmethod
    def online_filter(recorder):
        tags = recorder.list_tags()
        if tags.get(OnlineManager.ONLINE_KEY, OnlineManager.OFFLINE_TAG) == OnlineManager.ONLINE_TAG:
            return True
        return False

    @staticmethod
    def next_online_filter(recorder):
        tags = recorder.list_tags()
        if tags.get(OnlineManager.ONLINE_KEY, OnlineManager.OFFLINE_TAG) == OnlineManager.NEXT_ONLINE_TAG:
            return True
        return False

    def list_online_model(self):
        """list the record of online model

        Returns
        -------
        dict
            {rid : recorder of the online model}
        """

        return self.tc.list_recorders(rec_filter_func=self.online_filter)

    def list_next_online_model(self):
        return self.tc.list_recorders(rec_filter_func=self.next_online_filter)

    def update_online_pred(self):
        """update all online model predictions to the latest day in Calendar"""
        mu = ModelUpdater(self.exp_name)
        cnt = mu.update_all_pred(self.online_filter)
        self.logger.info(f"Finish updating {cnt} online model predictions of {self.exp_name}.")


class RollingOnlineManager(OnlineManager):
    def __init__(self, experiment_name: str, rolling_gen: RollingGen, task_pool) -> None:
        super().__init__(experiment_name)
        self.ta = TimeAdjuster()
        self.rg = rolling_gen
        self.tm = TaskManager(task_pool=task_pool)
        self.logger = get_module_logger("RollingOnlineManager")

    def prepare_new_models(self):
        """prepare(train) new models based on online model"""
        latest_records = self.tc.list_latest_recorders(self.online_filter)  # if we need online_filter here?
        max_test = self.tc.latest_time(latest_records)
        calendar_latest = self.ta.last_date()
        if self.ta.cal_interval(calendar_latest, max_test[0]) > self.rg.step:
            old_tasks = []
            for rid, rec in latest_records.items():
                task = self.tc.get_task(rec)
                test_begin = task["dataset"]["kwargs"]["segments"]["test"][0]
                # modify the test segment to generate new tasks
                task["dataset"]["kwargs"]["segments"]["test"] = (test_begin, calendar_latest)
                old_tasks.append(task)
            new_tasks = task_generator(old_tasks, self.rg)
            self.tm.create_task(new_tasks)
            run_task(task_train, self.tm.task_pool, experiment_name=self.exp_name)
            self.logger.info(f"Finished prepare {len(new_tasks)} new models.")
            return new_tasks
        self.logger.info("No need to prepare any new models.")
        return []

    def prepare_signals(self):
        # prepare the signals of today
        pass

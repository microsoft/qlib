from qlib.workflow import R
import pandas as pd
from tqdm.auto import tqdm
from typing import Union
from typing import Callable

from qlib import get_module_logger


class TaskCollector:
    """
    Collect the record (or its results) of the tasks
    """

    def __init__(self, experiment_name: str) -> None:
        self.exp_name = experiment_name
        self.exp = R.get_exp(experiment_name=experiment_name)
        self.logger = get_module_logger("TaskCollector")

    def list_recorders(self, rec_filter_func=None):

        recs = self.exp.list_recorders()
        recs_flt = {}
        for rid, rec in recs.items():
            if rec_filter_func is None or rec_filter_func(rec):
                recs_flt[rid] = rec

        return recs_flt

    def list_recorders_by_task(self, task_filter_func=None):
        def rec_filter(recorder):
            return task_filter_func(self.get_task(recorder))

        return self.list_recorders(rec_filter)

    def list_latest_recorders(self, rec_filter_func=None):
        recs_flt = self.list_recorders(rec_filter_func)
        max_test = self.latest_time(recs_flt)
        latest_rec = {}
        for rid, rec in recs_flt.items():
            if self.get_task(rec)["dataset"]["kwargs"]["segments"]["test"] == max_test:
                latest_rec[rid] = rec
        return latest_rec

    def get_recorder_by_id(self, recorder_id):
        return self.exp.get_recorder(recorder_id, create=False)

    def get_task(self, recorder):
        if isinstance(recorder, str):
            recorder = self.get_recorder_by_id(recorder_id=recorder)
        try:
            task = recorder.load_object("task")
        except OSError:
            raise OSError(f"Can't find task in {recorder.info['id']}, have you trained with model.trainer.task_train?")
        return task

    def latest_time(self, recorders):
        if len(recorders) == 0:
            raise Exception(f"Can't find any recorder in {self.exp_name}")
        max_test = max(self.get_task(rec)["dataset"]["kwargs"]["segments"]["test"] for rec in recorders.values())
        return max_test


class RollingCollector(TaskCollector):
    """
    Collect the record results of the rolling tasks
    """

    def __init__(
        self,
        experiment_name: str,
    ) -> None:
        super().__init__(experiment_name)
        self.logger = get_module_logger("RollingCollector")

    def collect_rolling_predictions(self, get_key_func, rec_filter_func=None):
        """For rolling tasks, the predictions will be in the diffierent recorder.
        To collect and concat the predictions of one rolling task, get_key_func will help this method see which group a recorder will be.

        Parameters
        ----------
        get_key_func : Callable[dict,str]
            a function that get task config and return its group str
        rec_filter_func : Callable[Recorder,bool], optional
            a function that decide whether filter a recorder, by default None

        Returns
        -------
        dict
            a dict of {group: predictions}
        """

        # filter records
        recs_flt = self.list_recorders(rec_filter_func)

        # group
        recs_group = {}
        for _, rec in recs_flt.items():
            task = self.get_task(rec)
            group_key = get_key_func(task)
            recs_group.setdefault(group_key, []).append(rec)

        # reduce group
        reduce_group = {}
        for k, rec_l in recs_group.items():
            pred_l = []
            for rec in rec_l:
                pred_l.append(rec.load_object("pred.pkl").iloc[:, 0])
            # Make sure the pred are sorted according to the rolling start time
            pred_l.sort(key=lambda pred: pred.index.get_level_values("datetime").min())
            pred = pd.concat(pred_l)
            # If there are duplicated predition, we use the latest perdiction
            pred = pred[~pred.index.duplicated(keep="last")]
            pred = pred.sort_index()
            reduce_group[k] = pred

        return reduce_group

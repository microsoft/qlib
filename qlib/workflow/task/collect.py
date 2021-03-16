from qlib.workflow import R
import pandas as pd
from typing import Union
from typing import Callable

from qlib import get_module_logger


class TaskCollector:
    """
    Collect the record results of the finished tasks with key and filter
    """

    def __init__(self, experiment_name: str) -> None:
        self.exp_name = experiment_name
        self.exp = R.get_exp(experiment_name=experiment_name)
        self.logger = get_module_logger("TaskCollector")

    def list_recorders(self, rec_filter_func=None):
        """"""
        recs = self.exp.list_recorders()
        recs_flt = {}
        for rid, rec in recs.items():
            if rec_filter_func is None or rec_filter_func(rec):
                recs_flt[rid] = rec

        return recs_flt

    def get_recorder_by_id(self, recorder_id):
        return self.exp.get_recorder(recorder_id, create=False)

    def list_recorders_by_task(self, task_filter_func):
        """[summary]

        Parameters
        ----------
        task_filter_func : [type], optional
            [description], by default None
        """

        def rec_filter_func(recorder):
            try:
                task = recorder.load_object("task")
            except OSError:
                raise OSError(
                    f"Can't find task in {recorder.info['id']}, have you trained with model.trainer.task_train?"
                )
            return task_filter_func(task)

        return self.list_recorders(rec_filter_func)

    def collect_predictions(
        self,
        get_key_func,
        task_filter_func=None,
    ):
        """
        Collect predictions using a filter and a key function.

        Parameters
        ----------
        experiment_name : str
        get_key_func : Callable[[dict], bool] -> Union[Number, str, tuple]
            get the key of a task when collect it
        filter_func : Callable[[dict], bool] -> bool
            to judge a task will be collected or not

        Returns
        -------
        dict
            the dict of predictions
        """
        recs_flt = self.list_recorders(task_filter_func=task_filter_func, only_have_task=True)

        # group
        recs_group = {}
        for _, rec in recs_flt.items():
            params = rec.task
            group_key = get_key_func(params)
            recs_group.setdefault(group_key, []).append(rec)

        # reduce group
        reduce_group = {}
        for k, rec_l in recs_group.items():
            pred_l = []
            for rec in rec_l:
                pred_l.append(rec.load_object("pred.pkl").iloc[:, 0])
            pred = pd.concat(pred_l).sort_index()
            reduce_group[k] = pred

        self.logger.info(f"Collect {len(reduce_group)} predictions in {self.exp_name}")
        return reduce_group

    def collect_latest_records(
        self,
        task_filter_func=None,
    ):
        """Collect latest recorders using a filter.

        Parameters
        ----------
        task_filter_func : Callable[[dict], bool], optional
            to judge a task will be collected or not, by default None

        Returns
        -------
        dict, tuple
            a dict of recorders and a tuple of test segments
        """
        recs_flt = self.list_recorders(task_filter_func=task_filter_func, only_have_task=True)

        if len(recs_flt) == 0:
            self.logger.warning("Can not collect any recorders...")
            return None, None
        max_test = max(rec.task["dataset"]["kwargs"]["segments"]["test"] for rec in recs_flt.values())

        latest_record = {}
        for rid, rec in recs_flt.items():
            if rec.task["dataset"]["kwargs"]["segments"]["test"] == max_test:
                latest_record[rid] = rec

        self.logger.info(f"Collect {len(latest_record)} latest records in {self.exp_name}")
        return latest_record, max_test

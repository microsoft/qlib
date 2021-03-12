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

    def list_recorders(self, rec_filter_func=None, task_filter_func=None, only_finished=True, only_have_task=False):
        """
        Return a dict of {rid:Recorder} by recorder filter and task filter. It is not necessary to use those filter. 
        If you don't train with "task_train", then there is no "task" which includes the task config.
        If there is a "task", then it will become rec.task which can be get simply.

        Parameters
        ----------
        rec_filter_func : Callable[[Recorder], bool], optional
            judge whether you need this recorder, by default None
        task_filter_func : Callable[[dict], bool], optional
            judge whether you need this task, by default None
        only_finished : bool, optional
            whether always use finished recorder, by default True
        only_have_task : bool, optional
            whether it is necessary to get the task config

        Returns
        -------
        dict
            a dict of {rid:Recorder}

        Raises
        ------
        OSError
            if you use a task filter, but there is no "task" which includes the task config
        """
        recs = self.exp.list_recorders()
        recs_flt = {}
        if task_filter_func is not None:
            only_have_task = True
        for rid, rec in recs.items():
            if (only_finished and rec.status == rec.STATUS_FI) or only_finished==False:
                if rec_filter_func is None or rec_filter_func(rec):
                    task = None
                    try:
                        task = rec.load_object("task")
                    except OSError:
                        pass
                    if task is None and only_have_task:
                        continue
                    if task_filter_func is None or task_filter_func(task):
                        rec.task = task
                        recs_flt[rid] = rec
                        
        return recs_flt

    def collect_predictions(
        self,
        get_key_func,
        task_filter_func=None,
    ):
        """

        Parameters
        ----------
        experiment_name : str
        get_key_func : function(task: dict) -> Union[Number, str, tuple]
            get the key of a task when collect it
        filter_func : function(task: dict) -> bool
            to judge a task will be collected or not

        Returns
        -------
        dict
            the dict of predictions
        """
        recs_flt = self.list_recorders(task_filter_func=task_filter_func,only_have_task=True)

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
        recs_flt = self.list_recorders(task_filter_func=task_filter_func,only_have_task=True)
        
        if len(recs_flt) == 0:
            self.logger.warning("Can not collect any recorders...")
            return None, None
        max_test = max(rec.task['dataset']['kwargs']['segments']['test'] for rec in recs_flt.values())

        latest_record = {}
        for rid, rec in recs_flt.items():
            if rec.task['dataset']['kwargs']['segments']['test'] == max_test:
                latest_record[rid] = rec
        
        self.logger.info(f"Collect {len(latest_record)} latest records in {self.exp_name}")
        return latest_record, max_test
        
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
        Return a dict of {rid:recorder} by recorder filter and task filter. It is not necessary to use those filter. 
        If you don't train with "task_train", then there is no "task.pkl" which includes the task config.
        If there is a "task.pkl", then it will become rec.task which can be get simply.

        Parameters
        ----------
        rec_filter_func : Callable[[MLflowRecorder], bool], optional
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
            a dict of {rid:recorder}

        Raises
        ------
        OSError
            if you use a task filter, but there is no "task.pkl" which includes the task config
        """
        recs = self.exp.list_recorders()
        # return all recorders if the filter is None and you don't need task
        if rec_filter_func==None and task_filter_func==None and only_have_task==False:
            return recs
        recs_flt = {}
        for rid, rec in recs.items():
            if (only_finished and rec.status == rec.STATUS_FI) or only_finished==False:
                if rec_filter_func is None or rec_filter_func(rec):
                    task = None
                    try:
                        task = rec.load_object("task.pkl")
                    except OSError:
                        if task_filter_func is not None:
                            raise OSError('Can not find "task.pkl" in your records, have you train with "task_train" method in qlib.model.trainer?')
                    if task is None and only_have_task:
                        continue
                    
                    if task_filter_func is None or task_filter_func(task):
                        rec.task = task
                        recs_flt[rid] = rec
                        
        return recs_flt

    def collect_predictions(
        self,
        get_key_func,
        filter_func=None,
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
        recs_flt = self.list_recorders(task_filter_func=filter_func)

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
        filter_func=None,
    ):
        recs_flt = self.list_recorders(task_filter_func=filter_func,only_have_task=True)

        max_test = max(rec.task['dataset']['kwargs']['segments']['test'] for rec in  recs_flt.values())

        latest_record = {}
        for rid, rec in recs_flt.items():
            if rec.task['dataset']['kwargs']['segments']['test'] == max_test:
                latest_record[rid] = rec
        
        self.logger.info(f"Collect {len(latest_record)} latest records in {self.exp_name}")
        return latest_record
        


class RollingCollector:
    """
    Rolling Models Ensemble based on (R)ecord

    This shares nothing with Ensemble
    """

    # TODO: speed up this class
    def __init__(self, get_key_func, flt_func=None):
        self.get_key_func = get_key_func  # get the key of a task based on task config
        self.flt_func = flt_func  # determine whether a task can be retained based on task config

    def __call__(self, exp_name) -> Union[pd.Series, dict]:
        # TODO;
        # Should we split the scripts into several sub functions?
        exp = R.get_exp(experiment_name=exp_name)

        # filter records
        recs = exp.list_recorders()

        recs_flt = {}
        for rid, rec in tqdm(recs.items(), desc="Loading data"):
            params = rec.load_object("task.pkl")
            if rec.status == rec.STATUS_FI:
                if self.flt_func is None or self.flt_func(params):
                    rec.params = params
                    recs_flt[rid] = rec

        # group
        recs_group = {}
        for _, rec in recs_flt.items():
            params = rec.params
            group_key = self.get_key_func(params)
            recs_group.setdefault(group_key, []).append(rec)

        # reduce group
        reduce_group = {}
        for k, rec_l in recs_group.items():
            pred_l = []
            for rec in rec_l:
                pred_l.append(rec.load_object("pred.pkl").iloc[:, 0])
            pred = pd.concat(pred_l).sort_index()
            reduce_group[k] = pred

        return reduce_group

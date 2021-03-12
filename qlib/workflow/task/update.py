from typing import Union
from qlib.workflow import R
from tqdm.auto import tqdm
from qlib.data import D
import pandas as pd
from qlib.utils import init_instance_by_config
from qlib import get_module_logger
from qlib.workflow import R


class ModelUpdater:
    """
    The model updater to re-train model or update predictions
    """

    ONLINE_TAG = "online_model"
    ONLINE_TAG_TRUE = "True"
    ONLINE_TAG_FALSE = "False"

    def __init__(self, experiment_name: str) -> None:
        """ModelUpdater needs experiment name to find the records

        Parameters
        ----------
        experiment_name : str
            experiment name string
        """
        self.exp_name = experiment_name
        self.exp = R.get_exp(experiment_name=experiment_name)
        self.logger = get_module_logger("ModelUpdater")

    def set_online_model(self, rid: str):
        """online model will be identified at the tags of the record

        Parameters
        ----------
        rid : str
            the id of a record
        """
        rec = self.exp.get_recorder(recorder_id=rid)
        rec.set_tags(**{self.ONLINE_TAG: self.ONLINE_TAG_TRUE})

    def cancel_online_model(self, rid: str):
        rec = self.exp.get_recorder(recorder_id=rid)
        rec.set_tags(**{self.ONLINE_TAG: self.ONLINE_TAG_FALSE})

    def cancel_all_online_model(self):
        recs = self.exp.list_recorders()
        for rid, rec in recs.items():
            self.cancel_online_model(rid)

    def reset_online_model(self, rids: Union[str, list]):
        """cancel all online model and reset the given model to online model

        Parameters
        ----------
        rids : Union[str, list]
            the name of a record or the list of the name of records
        """
        self.cancel_all_online_model()
        if isinstance(rids, str):
            rids = [rids]
        for rid in rids:
            self.set_online_model(rid)

    def update_pred(self, rid: str):
        """update predictions to the latest day in Calendar based on rid

        Parameters
        ----------
        rid : str
            the id of the record
        """
        rec = self.exp.get_recorder(recorder_id=rid)
        old_pred = rec.load_object("pred.pkl")
        last_end = old_pred.index.get_level_values("datetime").max()
        task_config = rec.load_object("task")

        # updated to the latest trading day
        cal = D.calendar(start_time=last_end + pd.Timedelta(days=1), end_time=None)

        if len(cal) == 0:
            self.logger.info(f"All prediction in {rid} of {self.exp_name} are latest. No need to update.")
            return

        start_time, end_time = cal[0], cal[-1]
        task_config["dataset"]["kwargs"]["segments"]["test"] = (start_time, end_time)
        task_config["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"] = end_time

        dataset = init_instance_by_config(task_config["dataset"])

        model = rec.load_object("params.pkl")
        new_pred = model.predict(dataset)

        cb_pred = pd.concat([old_pred, new_pred.to_frame("score")], axis=0)
        cb_pred = cb_pred.sort_index()

        rec.save_objects(**{"pred.pkl": cb_pred})

        self.logger.info(f"Finish updating new {new_pred.shape[0]} predictions in {rid} of {self.exp_name}.")

    def update_all_pred(self, filter_func=None):
        """update all predictions in this experiment after filter.

            An example of filter function:

            .. code-block:: python

                def record_filter(record):
                    task_config = record.load_object("task")
                    if task_config["model"]["class"]=="LGBModel":
                        return True
                    return False

        Parameters
        ----------
        filter_func : function, optional
            the filter function to decide whether this record will be updated, by default None

        Returns
        ----------
        cnt: int
            the count of updated record

        """
        cnt = 0
        recs = self.exp.list_recorders()
        for rid, rec in recs.items():
            if rec.status == rec.STATUS_FI:
                if filter_func != None and filter_func(rec) == False:
                    # records that should be filtered out
                    continue
                self.update_pred(rid)
                cnt += 1
        return cnt

    def online_filter(self, record):
        tags = record.list_tags()
        if tags[self.ONLINE_TAG] == self.ONLINE_TAG_TRUE:
            return True
        return False

    def update_online_pred(self):
        """update all online model predictions to the latest day in Calendar."""
        cnt = self.update_all_pred(self.online_filter)
        self.logger.info(f"Finish updating {cnt} online model predictions of {self.exp_name}.")

    def list_online_model(self):
        recs = self.exp.list_recorders()
        online_rec = {}
        for rid, rec in recs.items():
            if self.online_filter(rec):
                online_rec[rid] = rec
        return online_rec

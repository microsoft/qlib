from typing import Union, List
from qlib.workflow import R
from tqdm.auto import tqdm
from qlib.data import D
import pandas as pd
from qlib.utils import init_instance_by_config
from qlib import get_module_logger
from qlib.workflow import R
from qlib.model.trainer import task_train
from qlib.workflow.recorder import Recorder
from qlib.workflow.task.collect import TaskCollector


class ModelUpdater(TaskCollector):
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

    def set_online_model(self, recorder: Union[str, Recorder]):
        """online model will be identified at the tags of the record

        Parameters
        ----------
        recorder: Union[str,Recorder]
            the id of a Recorder or the Recorder instance
        """
        if isinstance(recorder, str):
            recorder = self.exp.get_recorder(recorder_id=recorder)
        recorder.set_tags(**{ModelUpdater.ONLINE_TAG: ModelUpdater.ONLINE_TAG_TRUE})

    def cancel_online_model(self, recorder: Union[str, Recorder]):
        if isinstance(recorder, str):
            recorder = self.exp.get_recorder(recorder_id=recorder)
        recorder.set_tags(**{ModelUpdater.ONLINE_TAG: ModelUpdater.ONLINE_TAG_FALSE})

    def cancel_all_online_model(self):
        recs = self.exp.list_recorders()
        for rid, rec in recs.items():
            self.cancel_online_model(rec)

    def reset_online_model(self, recorders: List[Union[str, Recorder]]):
        """cancel all online model and reset the given model to online model

        Parameters
        ----------
        recorders: List[Union[str,Recorder]]
            the list of the id of a Recorder or the Recorder instance
        """
        self.cancel_all_online_model()
        for rec_or_rid in recorders:
            self.set_online_model(rec_or_rid)

    def update_pred(self, recorder: Union[str, Recorder]):
        """update predictions to the latest day in Calendar based on rid

        Parameters
        ----------
        recorder: Union[str,Recorder]
            the id of a Recorder or the Recorder instance
        """
        if isinstance(recorder, str):
            recorder = self.exp.get_recorder(recorder_id=recorder)
        old_pred = recorder.load_object("pred.pkl")
        last_end = old_pred.index.get_level_values("datetime").max()
        task_config = recorder.load_object("task")  # recorder.task

        # updated to the latest trading day
        cal = D.calendar(start_time=last_end + pd.Timedelta(days=1), end_time=None)

        if len(cal) == 0:
            self.logger.info(
                f"The prediction in {recorder.info['id']} of {self.exp_name} are latest. No need to update."
            )
            return

        start_time, end_time = cal[0], cal[-1]
        task_config["dataset"]["kwargs"]["segments"]["test"] = (start_time, end_time)
        task_config["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"] = end_time

        dataset = init_instance_by_config(task_config["dataset"])

        model = recorder.load_object("params.pkl")
        new_pred = model.predict(dataset)

        cb_pred = pd.concat([old_pred, new_pred.to_frame("score")], axis=0)
        cb_pred = cb_pred.sort_index()

        recorder.save_objects(**{"pred.pkl": cb_pred})

        self.logger.info(
            f"Finish updating new {new_pred.shape[0]} predictions in {recorder.info['id']} of {self.exp_name}."
        )

    def update_all_pred(self, rec_filter_func=None):
        """update all predictions in this experiment after filter.

            An example of filter function:

            .. code-block:: python

                def rec_filter_func(recorder):
                    task_config = recorder.load_object("task")
                    if task_config["model"]["class"]=="LGBModel":
                        return True
                    return False

        Parameters
        ----------
        rec_filter_func : Callable[[Recorder], bool], optional
            the filter function to decide whether this record will be updated, by default None

        Returns
        ----------
        cnt: int
            the count of updated record

        """
        recs = self.list_recorders(rec_filter_func=rec_filter_func, only_have_task=True)
        for rid, rec in recs.items():
            self.update_pred(rec)
        return len(recs)

    def online_filter(self, recorder):
        tags = recorder.list_tags()
        if tags.get(ModelUpdater.ONLINE_TAG, ModelUpdater.ONLINE_TAG_FALSE) == ModelUpdater.ONLINE_TAG_TRUE:
            return True
        return False

    def update_online_pred(self):
        """update all online model predictions to the latest day in Calendar."""
        cnt = self.update_all_pred(self.online_filter)
        self.logger.info(f"Finish updating {cnt} online model predictions of {self.exp_name}.")

    def list_online_model(self):
        """list the record of online model

        Returns
        -------
        dict
            {rid : recorder of the online model}
        """

        return self.list_recorders(rec_filter_func=self.online_filter)

from typing import Union, List
from qlib.workflow import R
from qlib.data import D
import pandas as pd
from qlib import get_module_logger
from qlib.workflow import R
from qlib.model.trainer import task_train
from qlib.workflow.recorder import Recorder
from qlib.workflow.task.collect import TaskCollector


class ModelUpdater:
    """
    The model updater to update model results in new data.
    """

    def __init__(self, experiment_name: str) -> None:
        """ModelUpdater needs experiment name to find the records

        Parameters
        ----------
        experiment_name : str
            experiment name string
        """
        self.exp_name = experiment_name
        self.logger = get_module_logger("ModelUpdater")
        self.tc = TaskCollector(experiment_name)

    def _reload_dataset(self, recorder, start_time, end_time):
        """reload dataset from pickle file

        Parameters
        ----------
        recorder : Recorder
            the instance of the Recorder
        start_time : Timestamp
            the start time you want to load
        end_time : Timestamp
            the end time you want to load

        Returns
        -------
        Dataset
            the instance of Dataset
        """
        segments = {"test": (start_time, end_time)}

        dataset = recorder.load_object("dataset")
        datahandler = recorder.load_object("datahandler")

        datahandler.conf_data(**{"start_time": start_time, "end_time": end_time})
        dataset.setup_data(handler=datahandler, segments=segments)
        datahandler.init(datahandler.IT_LS)
        return dataset

    def update_pred(self, recorder: Recorder):
        """update predictions to the latest day in Calendar based on rid

        Parameters
        ----------
        recorder: Union[str,Recorder]
            the id of a Recorder or the Recorder instance
        """
        old_pred = recorder.load_object("pred.pkl")
        last_end = old_pred.index.get_level_values("datetime").max()

        # updated to the latest trading day
        cal = D.calendar(start_time=last_end + pd.Timedelta(days=1), end_time=None)

        if len(cal) == 0:
            self.logger.info(
                f"The prediction in {recorder.info['id']} of {self.exp_name} are latest. No need to update."
            )
            return

        start_time, end_time = cal[0], cal[-1]

        dataset = self._reload_dataset(recorder, start_time, end_time)

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

                def record_filter(record):
                    task_config = record.load_object("task")
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
        recs = self.tc.list_recorders(rec_filter_func=rec_filter_func)
        for rid, rec in recs.items():
            self.update_pred(rec)
        return len(recs)

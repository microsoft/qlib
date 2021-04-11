from typing import Union, List
from qlib.data.dataset import DatasetH
from qlib.workflow import R
from qlib.data import D
import pandas as pd
from qlib import get_module_logger
from qlib.workflow import R
from qlib.model import Model
from qlib.model.trainer import task_train
from qlib.workflow.recorder import Recorder
from qlib.workflow.task.utils import list_recorders
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import DatasetH
from abc import ABCMeta, abstractmethod
from qlib.utils import get_date_by_shift


class RMDLoader:
    """
    Recorder Model Dataset Loader
    """

    def __init__(self, rec: Recorder):
        self.rec = rec

    def get_dataset(self, start_time, end_time, segments=None) -> DatasetH:
        """
        load, config and setup dataset.

        This dataset is for inferene

        Parameters
        ----------
        start_time :
            the start_time of underlying data
        end_time :
            the end_time of underlying data
        segments : dict
            the segments config for dataset
            Due to the time series dataset (TSDatasetH), the test segments maybe different from start_time and end_time
        """
        if segments is None:
            segments = {"test": (start_time, end_time)}
        dataset: DatasetH = self.rec.load_object("dataset")
        dataset.config(handler_kwargs={"start_time": start_time, "end_time": end_time}, segments=segments)
        dataset.setup_data(handler_kwargs={"init_type": DataHandlerLP.IT_LS})
        return dataset

    def get_model(self) -> Model:
        return self.rec.load_object("params.pkl")


class RecordUpdater(metaclass=ABCMeta):
    """
    Updata a specific recorders
    """

    def __init__(self, record: Recorder, *args, **kwargs):
        self.record = record

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Update info for specific recorder
        """
        ...


class PredUpdater(RecordUpdater):
    """
    Update the prediction in the Recorder
    """

    LATEST = "__latest"

    def __init__(self, record: Recorder, to_date=LATEST, hist_ref: int = 0, freq="day"):
        """
        Parameters
        ----------
        record : Recorder
        to_date :
            update to prediction to the `to_date`
        hist_ref : int
            Sometimes, the dataset will have historical depends.
            Leave the problem to user to set the length of historical dependancy
            NOTE: the start_time is not included in the hist_ref
            # TODO: automate this step in the future.
        """
        super().__init__(record=record)

        self.to_date = to_date
        self.hist_ref = hist_ref
        self.freq = freq
        self.rmdl = RMDLoader(rec=record)

        if to_date == self.LATEST:
            to_date = D.calendar(freq=freq)[-1]
        self.to_date = pd.Timestamp(to_date)
        self.old_pred = record.load_object("pred.pkl")
        self.last_end = self.old_pred.index.get_level_values("datetime").max()

    def prepare_data(self) -> DatasetH:
        """
        # Load dataset

        Seperating this function will make it easier to reuse the dataset
        """
        start_time_buffer = get_date_by_shift(self.last_end, -self.hist_ref + 1, clip_shift=False, freq=self.freq)
        start_time = get_date_by_shift(self.last_end, 1, freq=self.freq)
        seg = {"test": (start_time, self.to_date)}
        dataset = self.rmdl.get_dataset(start_time=start_time_buffer, end_time=self.to_date, segments=seg)
        return dataset

    def update(self, dataset: DatasetH = None):
        """
        update the precition in a recorder
        """
        # FIXME: the problme below is not solved
        # The model dumped on GPU instances can not be loaded on CPU instance. Follow exception will raised
        # RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.

        # load dataset
        if dataset is None:
            # For reusing the dataset
            dataset = self.prepare_data()

        # Load model
        model = self.rmdl.get_model()

        new_pred = model.predict(dataset)

        cb_pred = pd.concat([self.old_pred, new_pred.to_frame("score")], axis=0)
        cb_pred = cb_pred.sort_index()

        self.record.save_objects(**{"pred.pkl": cb_pred})

        get_module_logger(self.__class__.__name__).info(
            f"Finish updating new {new_pred.shape[0]} predictions in {self.record.info['id']}."
        )


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
        self.logger = get_module_logger(self.__class__.__name__)

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
        dataset.config(handler_kwargs={"start_time": start_time, "end_time": end_time}, segments=segments)
        dataset.setup_data(handler_kwargs={"init_type": DataHandlerLP.IT_LS})
        return dataset

    def update_pred(self, recorder: Recorder, frequency="day"):
        """update predictions to the latest day in Calendar based on rid

        Parameters
        ----------
        recorder: Union[str,Recorder]
            the id of a Recorder or the Recorder instance
        """
        old_pred = recorder.load_object("pred.pkl")
        last_end = old_pred.index.get_level_values("datetime").max()

        # updated to the latest trading day
        if frequency == "day":
            cal = D.calendar(start_time=last_end + pd.Timedelta(days=1), end_time=None)
        else:
            raise NotImplementedError("Now `ModelUpdater` only support update daily frequency prediction")

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
        recs = list_recorders(self.exp_name, rec_filter_func=rec_filter_func)
        for rid, rec in recs.items():
            self.update_pred(rec)
        return len(recs)

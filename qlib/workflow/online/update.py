# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Updater is a module to update artifacts such as predictions when the stock data is updating.
"""

from abc import ABCMeta, abstractmethod

import pandas as pd
from qlib import get_module_logger
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model import Model
from qlib.utils import get_date_by_shift
from qlib.workflow.recorder import Recorder


class RMDLoader:
    """
    Recorder Model Dataset Loader
    """

    def __init__(self, rec: Recorder):
        self.rec = rec

    def get_dataset(self, start_time, end_time, segments=None) -> DatasetH:
        """
        Load, config and setup dataset.

        This dataset is for inference.

        Args:
            start_time :
                the start_time of underlying data
            end_time :
                the end_time of underlying data
            segments : dict
                the segments config for dataset
                Due to the time series dataset (TSDatasetH), the test segments maybe different from start_time and end_time

        Returns:
            DatasetH: the instance of DatasetH

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
    Update a specific recorders
    """

    def __init__(self, record: Recorder, *args, **kwargs):
        self.record = record
        self.logger = get_module_logger(self.__class__.__name__)

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

    def __init__(self, record: Recorder, to_date=None, hist_ref: int = 0, freq="day"):
        """
        Init PredUpdater.

        Args:
            record : Recorder
            to_date :
                update to prediction to the `to_date`
            hist_ref : int
                Sometimes, the dataset will have historical depends.
                Leave the problem to users to set the length of historical dependency

                .. note::

                    the start_time is not included in the hist_ref

        """
        # TODO: automate this hist_ref in the future.
        super().__init__(record=record)

        self.to_date = to_date
        self.hist_ref = hist_ref
        self.freq = freq
        self.rmdl = RMDLoader(rec=record)

        if to_date == None:
            to_date = D.calendar(freq=freq)[-1]
        self.to_date = pd.Timestamp(to_date)
        self.old_pred = record.load_object("pred.pkl")
        self.last_end = self.old_pred.index.get_level_values("datetime").max()

    def prepare_data(self) -> DatasetH:
        """
        Load dataset

        Separating this function will make it easier to reuse the dataset

        Returns:
            DatasetH: the instance of DatasetH
        """
        start_time_buffer = get_date_by_shift(self.last_end, -self.hist_ref + 1, clip_shift=False, freq=self.freq)
        start_time = get_date_by_shift(self.last_end, 1, freq=self.freq)
        seg = {"test": (start_time, self.to_date)}
        dataset = self.rmdl.get_dataset(start_time=start_time_buffer, end_time=self.to_date, segments=seg)
        return dataset

    def update(self, dataset: DatasetH = None):
        """
        Update the prediction in a recorder.

        Args:
            DatasetH: the instance of DatasetH. None for reprepare.
        """
        # FIXME: the problem below is not solved
        # The model dumped on GPU instances can not be loaded on CPU instance. Follow exception will raised
        # RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.
        # https://github.com/pytorch/pytorch/issues/16797

        if self.last_end >= self.to_date:
            self.logger.info(
                f"The prediction in {self.record.info['id']} are latest ({self.last_end}). No need to update to {self.to_date}."
            )
            return

        # load dataset
        if dataset is None:
            # For reusing the dataset
            dataset = self.prepare_data()

        # Load model
        model = self.rmdl.get_model()

        new_pred: pd.Series = model.predict(dataset)

        cb_pred = pd.concat([self.old_pred, new_pred.to_frame("score")], axis=0)
        cb_pred = cb_pred.sort_index()

        self.record.save_objects(**{"pred.pkl": cb_pred})

        self.logger.info(f"Finish updating new {new_pred.shape[0]} predictions in {self.record.info['id']}.")

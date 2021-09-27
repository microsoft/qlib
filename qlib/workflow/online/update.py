# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Updater is a module to update artifacts such as predictions when the stock data is updating.
"""

from abc import ABCMeta, abstractmethod

import pandas as pd
from qlib import get_module_logger
from qlib.data import D
from qlib.data.dataset import Dataset, DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model import Model
from qlib.utils import get_date_by_shift
from qlib.workflow.recorder import Recorder
from qlib.workflow.record_temp import SignalRecord


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


class DSBasedUpdater(RecordUpdater, metaclass=ABCMeta):
    """
    Dataset-Based Updater
    - Provding updating feature for Updating data based on Qlib Dataset

    Assumption
    - Based on Qlib dataset
    - The data to be updated is a multi-level index pd.DataFrame. For example label , prediction.

                                 LABEL0
        datetime   instrument
        2021-05-10 SH600000    0.006965
                   SH600004    0.003407
        ...                         ...
        2021-05-28 SZ300498    0.015748
                   SZ300676   -0.001321
    """

    def __init__(self, record: Recorder, to_date=None, hist_ref: int = 0, freq="day", fname="pred.pkl"):
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
        self.fname = fname
        self.rmdl = RMDLoader(rec=record)

        latest_date = D.calendar(freq=freq)[-1]
        if to_date == None:
            to_date = latest_date
        to_date = pd.Timestamp(to_date)

        if to_date >= latest_date:
            self.logger.warning(
                f"The given `to_date`({to_date}) is later than `latest_date`({latest_date}). So `to_date` is clipped to `latest_date`."
            )
            to_date = latest_date
        self.to_date = to_date
        # FIXME: it will raise error when running routine with delay trainer
        # should we use another prediction updater for delay trainer?
        self.old_data: pd.DataFrame = record.load_object(fname)

        # dropna is for being compatible to some data with future information(e.g. label)
        # The recent label data should be updated together
        self.last_end = self.old_data.dropna().index.get_level_values("datetime").max()

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
        Update the data in a recorder.

        Args:
            DatasetH: the instance of DatasetH. None for reprepare.
        """
        # FIXME: the problem below is not solved
        # The model dumped on GPU instances can not be loaded on CPU instance. Follow exception will raised
        # RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.
        # https://github.com/pytorch/pytorch/issues/16797

        if self.last_end >= self.to_date:
            self.logger.info(
                f"The data in {self.record.info['id']} are latest ({self.last_end}). No need to update to {self.to_date}."
            )
            return

        # load dataset
        if dataset is None:
            # For reusing the dataset
            dataset = self.prepare_data()

        self.record.save_objects(**{self.fname: self.get_update_data(dataset)})

    @abstractmethod
    def get_update_data(self, dataset: Dataset) -> pd.DataFrame:
        """
        return the updated data based on the given dataset

        The difference between `get_update_data` and `update`
        - `update_date` only include some data specific feature
        - `update` include some general routine steps(e.g. prepare dataset, checking)
        """
        ...


class PredUpdater(DSBasedUpdater):
    """
    Update the prediction in the Recorder
    """

    def get_update_data(self, dataset: Dataset) -> pd.DataFrame:
        # Load model
        model = self.rmdl.get_model()
        new_pred: pd.Series = model.predict(dataset)

        cb_pred = pd.concat([self.old_data, new_pred.to_frame("score")], axis=0)
        cb_pred = cb_pred.sort_index()
        self.logger.info(f"Finish updating new {new_pred.shape[0]} predictions in {self.record.info['id']}.")
        return cb_pred


class LabelUpdater(DSBasedUpdater):
    """
    Update the label in the recorder

    Assumption
    - The label is generated from record_temp.SignalRecord.
    """

    def __init__(self, record: Recorder, to_date=None, **kwargs):
        super().__init__(record, to_date=to_date, fname="label.pkl", **kwargs)

    def get_update_data(self, dataset: Dataset) -> pd.DataFrame:
        new_label = SignalRecord.generate_label(dataset)
        cb_data = pd.concat([self.old_data, new_label], axis=0)
        cb_data = cb_data[~cb_data.index.duplicated(keep="last")].sort_index()
        return cb_data

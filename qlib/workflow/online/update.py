# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Updater is a module to update artifacts such as predictions when the stock data is updating.
"""

from abc import ABCMeta, abstractmethod
from typing import Optional

import pandas as pd
from qlib import get_module_logger
from qlib.data import D
from qlib.data.dataset import Dataset, DatasetH, TSDatasetH
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

    def get_dataset(
        self, start_time, end_time, segments=None, unprepared_dataset: Optional[DatasetH] = None
    ) -> DatasetH:
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
            unprepared_dataset: Optional[DatasetH]
                if user don't want to load dataset from recorder, please specify user's dataset

        Returns:
            DatasetH: the instance of DatasetH

        """
        if segments is None:
            segments = {"test": (start_time, end_time)}
        if unprepared_dataset is None:
            dataset: DatasetH = self.rec.load_object("dataset")
        else:
            dataset = unprepared_dataset
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


class DSBasedUpdater(RecordUpdater, metaclass=ABCMeta):
    """
    Dataset-Based Updater

    - Providing updating feature for Updating data based on Qlib Dataset

    Assumption

    - Based on Qlib dataset
    - The data to be updated is a multi-level index pd.DataFrame. For example label, prediction.

        .. code-block::

                                     LABEL0
            datetime   instrument
            2021-05-10 SH600000    0.006965
                       SH600004    0.003407
            ...                         ...
            2021-05-28 SZ300498    0.015748
                       SZ300676   -0.001321
    """

    def __init__(
        self,
        record: Recorder,
        to_date=None,
        from_date=None,
        hist_ref: Optional[int] = None,
        freq="day",
        fname="pred.pkl",
        loader_cls: type = RMDLoader,
    ):
        """
        Init PredUpdater.

        Expected behavior in following cases:

        - if `to_date` is greater than the max date in the calendar, the data will be updated to the latest date
        - if there are data before `from_date` or after `to_date`, only the data between `from_date` and `to_date` are affected.

        Args:
            record : Recorder
            to_date :
                update to prediction to the `to_date`

                if to_date is None:

                    data will updated to the latest date.
            from_date :
                the update will start from `from_date`

                if from_date is None:

                    the updating will occur on the next tick after the latest data in historical data
            hist_ref : int
                Sometimes, the dataset will have historical depends.
                Leave the problem to users to set the length of historical dependency
                If user doesn't specify this parameter, Updater will try to load dataset to automatically determine the hist_ref

                .. note::

                    the start_time is not included in the `hist_ref`; So the `hist_ref` will be `step_len - 1` in most cases

            loader_cls : type
                the class to load the model and dataset

        """
        # TODO: automate this hist_ref in the future.
        super().__init__(record=record)

        self.to_date = to_date
        self.hist_ref = hist_ref
        self.freq = freq
        self.fname = fname
        self.rmdl = loader_cls(rec=record)

        latest_date = D.calendar(freq=freq)[-1]
        if to_date is None:
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
        if from_date is None:
            # dropna is for being compatible to some data with future information(e.g. label)
            # The recent label data should be updated together
            self.last_end = self.old_data.dropna().index.get_level_values("datetime").max()
        else:
            self.last_end = get_date_by_shift(from_date, -1, align="right")

    def prepare_data(self, unprepared_dataset: Optional[DatasetH] = None) -> DatasetH:
        """
        Load dataset
        - if unprepared_dataset is specified, then prepare the dataset directly
        - Otherwise,

        Separating this function will make it easier to reuse the dataset

        Returns:
            DatasetH: the instance of DatasetH
        """
        # automatically getting the historical dependency if not specified
        if self.hist_ref is None:
            dataset: DatasetH = self.record.load_object("dataset") if unprepared_dataset is None else unprepared_dataset
            # Special treatment of historical dependencies
            if isinstance(dataset, TSDatasetH):
                hist_ref = dataset.step_len - 1
            else:
                hist_ref = 0  # if only the lastest data is used, then only current data will be used and no historical data will be used
        else:
            hist_ref = self.hist_ref

        start_time_buffer = get_date_by_shift(
            self.last_end, -hist_ref + 1, clip_shift=False, freq=self.freq  # pylint: disable=E1130
        )
        start_time = get_date_by_shift(self.last_end, 1, freq=self.freq)
        seg = {"test": (start_time, self.to_date)}
        return self.rmdl.get_dataset(
            start_time=start_time_buffer, end_time=self.to_date, segments=seg, unprepared_dataset=unprepared_dataset
        )

    def update(self, dataset: DatasetH = None, write: bool = True, ret_new: bool = False) -> Optional[object]:
        """
        Parameters
        ----------
        dataset : DatasetH
            DatasetH: the instance of DatasetH. None for prepare it again.
        write : bool
            will the the write action be executed
        ret_new : bool
            will the updated data be returned

        Returns
        -------
        Optional[object]
            the updated dataset
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

        updated_data = self.get_update_data(dataset)

        if write:
            self.record.save_objects(**{self.fname: updated_data})
        if ret_new:
            return updated_data

    @abstractmethod
    def get_update_data(self, dataset: Dataset) -> pd.DataFrame:
        """
        return the updated data based on the given dataset

        The difference between `get_update_data` and `update`
        - `update_date` only include some data specific feature
        - `update` include some general routine steps(e.g. prepare dataset, checking)
        """


def _replace_range(data, new_data):
    dates = new_data.index.get_level_values("datetime")
    data = data.sort_index()
    data = data.drop(data.loc[dates.min() : dates.max()].index)
    cb_data = pd.concat([data, new_data], axis=0)
    cb_data = cb_data[~cb_data.index.duplicated(keep="last")].sort_index()
    return cb_data


class PredUpdater(DSBasedUpdater):
    """
    Update the prediction in the Recorder
    """

    def get_update_data(self, dataset: Dataset) -> pd.DataFrame:
        # Load model
        model = self.rmdl.get_model()
        new_pred: pd.Series = model.predict(dataset)
        data = _replace_range(self.old_data, new_pred.to_frame("score"))
        self.logger.info(f"Finish updating new {new_pred.shape[0]} predictions in {self.record.info['id']}.")
        return data


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
        cb_data = _replace_range(self.old_data.sort_index(), new_label)
        return cb_data

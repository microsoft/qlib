# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding=utf-8
import abc
import bisect
import logging
import warnings
from inspect import getfullargspec
from typing import Callable, Union, Tuple, List, Iterator, Optional

import pandas as pd
import numpy as np

from ...log import get_module_logger, TimeInspector
from ...data import D
from ...config import C
from ...utils import parse_config, transform_end_date, init_instance_by_config
from ...utils.serial import Serializable
from .utils import fetch_df_by_index
from pathlib import Path
from .loader import DataLoader

from . import processor as processor_module
from . import loader as data_loader_module


# TODO: A more general handler interface which does not relies on internal pd.DataFrame is needed.
class DataHandler(Serializable):
    """
    The steps to using a handler
    1. initialized data handler  (call by `init`).
    2. use the data.


    The data handler try to maintain a handler with 2 level.
    `datetime` & `instruments`.

    Any order of the index level can be supported (The order will be implied in the data).
    The order  <`datetime`, `instruments`> will be used when the dataframe index name is missed.

    Example of the data:
    The multi-index of the columns is optional.

    .. code-block:: python

                                feature                                                            label
                                $close     $volume  Ref($close, 1)  Mean($close, 3)  $high-$low  LABEL0
        datetime   instrument
        2010-01-04 SH600000    81.807068  17145150.0       83.737389        83.016739    2.741058  0.0032
                   SH600004    13.313329  11800983.0       13.313329        13.317701    0.183632  0.0042
                   SH600005    37.796539  12231662.0       38.258602        37.919757    0.970325  0.0289


    Tips for improving the performance of datahandler
    - Fetching data with `col_set=CS_RAW` will return the raw data and may avoid pandas from copying the data when calling `loc`
    """

    def __init__(
        self,
        instruments=None,
        start_time=None,
        end_time=None,
        data_loader: Union[dict, str, DataLoader] = None,
        init_data=True,
        fetch_orig=True,
    ):
        """
        Parameters
        ----------
        instruments :
            The stock list to retrive.
        start_time :
            start_time of the original data.
        end_time :
            end_time of the original data.
        data_loader : Union[dict, str, DataLoader]
            data loader to load the data.
        init_data :
            initialize the original data in the constructor.
        fetch_orig : bool
            Return the original data instead of copy if possible.
        """
        # Set logger
        self.logger = get_module_logger("DataHandler")

        # Setup data loader
        assert data_loader is not None  # to make start_time end_time could have None default value

        # what data source to load data
        self.data_loader = init_instance_by_config(
            data_loader,
            None if (isinstance(data_loader, dict) and "module_path" in data_loader) else data_loader_module,
            accept_types=DataLoader,
        )

        # what data to be loaded from data source
        # For IDE auto-completion.
        self.instruments = instruments
        self.start_time = start_time
        self.end_time = end_time

        self.fetch_orig = fetch_orig
        if init_data:
            with TimeInspector.logt("Init data"):
                self.setup_data()
        super().__init__()

    def config(self, **kwargs):
        """
        configuration of data.
        # what data to be loaded from data source

        This method will be used when loading pickled handler from dataset.
        The data will be initialized with different time range.

        """
        attr_list = {"instruments", "start_time", "end_time"}
        for k, v in kwargs.items():
            if k in attr_list:
                setattr(self, k, v)

        for attr in attr_list:
            if attr in kwargs:
                kwargs.pop(attr)

        super().config(**kwargs)

    def setup_data(self, enable_cache: bool = False):
        """
        Set Up the data in case of running initialization for multiple time

        It is responsible for maintaining following variable
        1) self._data

        Parameters
        ----------
        enable_cache : bool
            default value is false:

            - if `enable_cache` == True:

                the processed data will be saved on disk, and handler will load the cached data from the disk directly
                when we call `init` next time
        """
        # Setup data.
        # _data may be with multiple column index level. The outer level indicates the feature set name
        with TimeInspector.logt("Loading data"):
            self._data = self.data_loader.load(self.instruments, self.start_time, self.end_time)
        # TODO: cache

    CS_ALL = "__all"  # return all columns with single-level index column
    CS_RAW = "__raw"  # return raw data with multi-level index column

    def _fetch_df_by_col(self, df: pd.DataFrame, col_set: str) -> pd.DataFrame:
        if not isinstance(df.columns, pd.MultiIndex) or col_set == self.CS_RAW:
            return df
        elif col_set == self.CS_ALL:
            return df.droplevel(axis=1, level=0)
        else:
            return df.loc(axis=1)[col_set]

    def fetch(
        self,
        selector: Union[pd.Timestamp, slice, str] = slice(None, None),
        level: Union[str, int] = "datetime",
        col_set: Union[str, List[str]] = CS_ALL,
        squeeze: bool = False,
        proc_func: Callable = None,
    ) -> pd.DataFrame:
        """
        fetch data from underlying data source

        Parameters
        ----------
        selector : Union[pd.Timestamp, slice, str]
            describe how to select data by index
        level : Union[str, int]
            which index level to select the data
        col_set : Union[str, List[str]]

            - if isinstance(col_set, str):

                select a set of meaningful columns.(e.g. features, columns)

                if cal_set == CS_RAW:
                    the raw dataset will be returned.

            - if isinstance(col_set, List[str]):

                select several sets of meaningful columns, the returned data has multiple levels
        proc_func: Callable
            - Give a hook for processing data before fetching
            - An example to explain the necessity of the hook:
                - A Dataset learned some processors to process data which is related to data segmentation
                - It will apply them every time when preparing data.
                - The learned processor require the dataframe remains the same format when fitting and applying
                - However the data format will change according to the parameters.
                - So the processors should be applied to the underlayer data.

        squeeze : bool
            whether squeeze columns and index

        Returns
        -------
        pd.DataFrame.
        """
        if proc_func is None:
            df = self._data
        else:
            # FIXME: fetching by time first will be more friendly to `proc_func`
            # Copy in case of `proc_func` changing the data inplace....
            df = proc_func(fetch_df_by_index(self._data, selector, level, fetch_orig=self.fetch_orig).copy())

        # Fetch column  first will be more friendly to SepDataFrame
        df = self._fetch_df_by_col(df, col_set)
        df = fetch_df_by_index(df, selector, level, fetch_orig=self.fetch_orig)
        if squeeze:
            # squeeze columns
            df = df.squeeze()
            # squeeze index
            if isinstance(selector, (str, pd.Timestamp)):
                df = df.reset_index(level=level, drop=True)
        return df

    def get_cols(self, col_set=CS_ALL) -> list:
        """
        get the column names

        Parameters
        ----------
        col_set : str
            select a set of meaningful columns.(e.g. features, columns)

        Returns
        -------
        list:
            list of column names
        """
        df = self._data.head()
        df = self._fetch_df_by_col(df, col_set)
        return df.columns.to_list()

    def get_range_selector(self, cur_date: Union[pd.Timestamp, str], periods: int) -> slice:
        """
        get range selector by number of periods

        Args:
            cur_date (pd.Timestamp or str): current date
            periods (int): number of periods
        """
        trading_dates = self._data.index.unique(level="datetime")
        cur_loc = trading_dates.get_loc(cur_date)
        pre_loc = cur_loc - periods + 1
        if pre_loc < 0:
            warnings.warn("`periods` is too large. the first date will be returned.")
            pre_loc = 0
        ref_date = trading_dates[pre_loc]
        return slice(ref_date, cur_date)

    def get_range_iterator(
        self, periods: int, min_periods: Optional[int] = None, **kwargs
    ) -> Iterator[Tuple[pd.Timestamp, pd.DataFrame]]:
        """
        get a iterator of sliced data with given periods

        Args:
            periods (int): number of periods.
            min_periods (int): minimum periods for sliced dataframe.
            kwargs (dict): will be passed to `self.fetch`.
        """
        trading_dates = self._data.index.unique(level="datetime")
        if min_periods is None:
            min_periods = periods
        for cur_date in trading_dates[min_periods:]:
            selector = self.get_range_selector(cur_date, periods)
            yield cur_date, self.fetch(selector, **kwargs)


class DataHandlerLP(DataHandler):
    """
    DataHandler with **(L)earnable (P)rocessor**

    Tips to improving the performance of data handler
    - To reduce the memory cost
        - `drop_raw=True`: this will modify the data inplace on raw data;
    """

    # data key
    DK_R = "raw"
    DK_I = "infer"
    DK_L = "learn"

    # process type
    PTYPE_I = "independent"
    # - self._infer will be processed by infer_processors
    # - self._learn will be processed by learn_processors
    PTYPE_A = "append"
    # - self._infer will be processed by infer_processors
    # - self._learn will be processed by infer_processors + learn_processors
    #   - (e.g. self._infer processed by learn_processors )

    def __init__(
        self,
        instruments=None,
        start_time=None,
        end_time=None,
        data_loader: Union[dict, str, DataLoader] = None,
        infer_processors=[],
        learn_processors=[],
        process_type=PTYPE_A,
        drop_raw=False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        infer_processors : list
            - list of <description info> of processors to generate data for inference

            - example of <description info>:

            .. code-block::

                1) classname & kwargs:
                    {
                        "class": "MinMaxNorm",
                        "kwargs": {
                            "fit_start_time": "20080101",
                            "fit_end_time": "20121231"
                        }
                    }
                2) Only classname:
                    "DropnaFeature"
                3) object instance of Processor

        learn_processors : list
            similar to infer_processors, but for generating data for learning models

        process_type: str
            PTYPE_I = 'independent'

            - self._infer will processed by infer_processors

            - self._learn will be processed by learn_processors

            PTYPE_A = 'append'

            - self._infer will processed by infer_processors

            - self._learn will be processed by infer_processors + learn_processors

              - (e.g. self._infer processed by learn_processors )
        drop_raw: bool
            Whether to drop the raw data
        """

        # Setup preprocessor
        self.infer_processors = []  # for lint
        self.learn_processors = []  # for lint
        for pname in "infer_processors", "learn_processors":
            for proc in locals()[pname]:
                getattr(self, pname).append(
                    init_instance_by_config(
                        proc,
                        None if (isinstance(proc, dict) and "module_path" in proc) else processor_module,
                        accept_types=processor_module.Processor,
                    )
                )

        self.process_type = process_type
        self.drop_raw = drop_raw
        super().__init__(instruments, start_time, end_time, data_loader, **kwargs)

    def get_all_processors(self):
        return self.infer_processors + self.learn_processors

    def fit(self):
        for proc in self.get_all_processors():
            with TimeInspector.logt(f"{proc.__class__.__name__}"):
                proc.fit(self._data)

    def fit_process_data(self):
        """
        fit and process data

        The input of the `fit` will be the output of the previous processor
        """
        self.process_data(with_fit=True)

    def process_data(self, with_fit: bool = False):
        """
        process_data data. Fun `processor.fit` if necessary

        Parameters
        ----------
        with_fit : bool
            The input of the `fit` will be the output of the previous processor
        """
        # data for inference
        _infer_df = self._data
        if len(self.infer_processors) > 0 and not self.drop_raw:  # avoid modifying the original  data
            _infer_df = _infer_df.copy()

        for proc in self.infer_processors:
            if not proc.is_for_infer():
                raise TypeError("Only processors usable for inference can be used in `infer_processors` ")
            with TimeInspector.logt(f"{proc.__class__.__name__}"):
                if with_fit:
                    proc.fit(_infer_df)
                _infer_df = proc(_infer_df)
        self._infer = _infer_df

        # data for learning
        if self.process_type == DataHandlerLP.PTYPE_I:
            _learn_df = self._data
        elif self.process_type == DataHandlerLP.PTYPE_A:
            # based on `infer_df` and append the processor
            _learn_df = _infer_df
        else:
            raise NotImplementedError(f"This type of input is not supported")

        if len(self.learn_processors) > 0:  # avoid modifying the original  data
            _learn_df = _learn_df.copy()
        for proc in self.learn_processors:
            with TimeInspector.logt(f"{proc.__class__.__name__}"):
                if with_fit:
                    proc.fit(_learn_df)
                _learn_df = proc(_learn_df)
        self._learn = _learn_df

        if self.drop_raw:
            del self._data

    def config(self, processor_kwargs: dict = None, **kwargs):
        """
        configuration of data.
        # what data to be loaded from data source

        This method will be used when loading pickled handler from dataset.
        The data will be initialized with different time range.

        """
        super().config(**kwargs)
        if processor_kwargs is not None:
            for processor in self.get_all_processors():
                processor.config(**processor_kwargs)

    # init type
    IT_FIT_SEQ = "fit_seq"  # the input of `fit` will be the output of the previous processor
    IT_FIT_IND = "fit_ind"  # the input of `fit` will be the original df
    IT_LS = "load_state"  # The state of the object has been load by pickle

    def setup_data(self, init_type: str = IT_FIT_SEQ, **kwargs):
        """
        Set up the data in case of running initialization for multiple time

        Parameters
        ----------
        init_type : str
            The type `IT_*` listed above.
        enable_cache : bool
            default value is false:

            - if `enable_cache` == True:

                the processed data will be saved on disk, and handler will load the cached data from the disk directly
                when we call `init` next time
        """
        # init raw data
        super().setup_data(**kwargs)

        with TimeInspector.logt("fit & process data"):
            if init_type == DataHandlerLP.IT_FIT_IND:
                self.fit()
                self.process_data()
            elif init_type == DataHandlerLP.IT_LS:
                self.process_data()
            elif init_type == DataHandlerLP.IT_FIT_SEQ:
                self.fit_process_data()
            else:
                raise NotImplementedError(f"This type of input is not supported")

        # TODO: Be able to cache handler data. Save the memory for data processing

    def _get_df_by_key(self, data_key: str = DK_I) -> pd.DataFrame:
        if data_key == self.DK_R and self.drop_raw:
            raise AttributeError(
                "DataHandlerLP has not attribute _data, please set drop_raw = False if you want to use raw data"
            )
        df = getattr(self, {self.DK_R: "_data", self.DK_I: "_infer", self.DK_L: "_learn"}[data_key])
        return df

    def fetch(
        self,
        selector: Union[pd.Timestamp, slice, str] = slice(None, None),
        level: Union[str, int] = "datetime",
        col_set=DataHandler.CS_ALL,
        data_key: str = DK_I,
        proc_func: Callable = None,
    ) -> pd.DataFrame:
        """
        fetch data from underlying data source

        Parameters
        ----------
        selector : Union[pd.Timestamp, slice, str]
            describe how to select data by index.
        level : Union[str, int]
            which index level to select the data.
        col_set : str
            select a set of meaningful columns.(e.g. features, columns).
        data_key : str
            the data to fetch:  DK_*.
        proc_func: Callable
            please refer to the doc of DataHandler.fetch

        Returns
        -------
        pd.DataFrame:
        """
        df = self._get_df_by_key(data_key)
        if proc_func is not None:
            # FIXME: fetch by time first will be more friendly to proc_func
            # Copy incase of `proc_func` changing the data inplace....
            df = proc_func(fetch_df_by_index(df, selector, level, fetch_orig=self.fetch_orig).copy())
        # Fetch column  first will be more friendly to SepDataFrame
        df = self._fetch_df_by_col(df, col_set)
        return fetch_df_by_index(df, selector, level, fetch_orig=self.fetch_orig)

    def get_cols(self, col_set=DataHandler.CS_ALL, data_key: str = DK_I) -> list:
        """
        get the column names

        Parameters
        ----------
        col_set : str
            select a set of meaningful columns.(e.g. features, columns).
        data_key : str
            the data to fetch:  DK_*.

        Returns
        -------
        list:
            list of column names
        """
        df = self._get_df_by_key(data_key).head()
        df = self._fetch_df_by_col(df, col_set)
        return df.columns.to_list()

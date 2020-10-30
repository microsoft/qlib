# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding=utf-8
import abc
import bisect
import logging
from typing import Union, Tuple, List, Iterator, Optional

import pandas as pd
import numpy as np

from ...log import get_module_logger, TimeInspector
from ...data import D
from ...config import C
from ...utils import parse_config, transform_end_date, init_instance_by_config
from ...utils.serial import Serializable
from .utils import get_level_index
from pathlib import Path
from .loader import DataLoader

from . import processor as processor_module
from . import loader as data_loader_module


# TODO: A more general handler interface which does not relies on internal pd.DataFrame is needed.
class DataHandler(Serializable):
    """
    The steps to using a handler
    1. initialized data handler  (call by `init`).
    2. use the data


    The data handler try to maintain a handler with 2 level.
    `datetime` & `instruments`.

    Any order of the index level can be suported(The order will implied in the data).
    The order  <`datetime`, `instruments`> will be used when the dataframe index name is missed.

    Example of the data:
    The multi-index of the columns is optional.
                             feature                                                            label
                              $close     $volume  Ref($close, 1)  Mean($close, 3)  $high-$low  LABEL0
    datetime   instrument
    2010-01-04 SH600000    81.807068  17145150.0       83.737389        83.016739    2.741058  0.0032
               SH600004    13.313329  11800983.0       13.313329        13.317701    0.183632  0.0042
               SH600005    37.796539  12231662.0       38.258602        37.919757    0.970325  0.0289

    """

    def __init__(
        self,
        instruments,
        start_time=None,
        end_time=None,
        data_loader: Tuple[dict, str, DataLoader] = None,
        init_data=True,
    ):
        # Set logger
        self.logger = get_module_logger("DataHandler")

        # Setup data loader
        assert data_loader is not None  # to make start_time end_time could have None default value
        self.data_loader = init_instance_by_config(data_loader, data_loader_module, accept_types=DataLoader)

        self.instruments = instruments
        self.start_time = start_time
        self.end_time = end_time
        if init_data:
            self.init()
        super().__init__()

    def init(self, enable_cache: bool = True):
        """
        initialize the data.
        In case of running intialization for multiple time, it will do nothing for the second time.

        It is responsible for maintaining following variable
        1) self._data

        Parameters
        ----------
        enable_cache : bool
            default value is false
            if `enable_cache` == True
                the processed data will be saved on disk, and handler will load the cached data from the disk directly
                when we call `init` next time
        """
        # Setup data.
        # _data may be with multiple column index level. The outer level indicates the feature set name
        self._data = self.data_loader.load(self.instruments, self.start_time, self.end_time)
        # TODO: cache

    def _fetch_df_by_index(
        self, df: pd.DataFrame, selector: Union[pd.Timestamp, slice, str, list], level: Union[str, int]
    ) -> pd.DataFrame:
        """
        fetch data from `data` with `selector` and `level`

        Parameters
        ----------
        selector : Union[pd.Timestamp, slice, str, list]
            selector
        level : Union[int, str]
            the level to use the selector
        """
        # Try to get the right index
        idx_slc = (selector, slice(None, None))
        if get_level_index(df, level) == 1:
            idx_slc = idx_slc[1], idx_slc[0]
        return df.loc(axis=0)[idx_slc]

    CS_ALL = "__all"

    def _fetch_df_by_col(self, df: pd.DataFrame, col_set: str) -> pd.DataFrame:
        if not isinstance(df.columns, pd.MultiIndex):
            return df
        elif col_set == self.CS_ALL:
            return df.droplevel(axis=1, level=0)
        else:
            return df.loc(axis=1)[col_set]

    def fetch(
        self,
        selector: Union[pd.Timestamp, slice, str],
        level: Union[str, int] = "datetime",
        col_set: Union[str, List[str]] = CS_ALL,
        squeeze: bool = False
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
            if isinstance(col_set, str):
                select a set of meaningful columns.(e.g. features, columns)
            if isinstance(col_set, List[str]):
                select several sets of meaningful columns, the returned data has multiple levels
        squeeze : bool
            whether squeeze columns and index

        Returns
        -------
        pd.DataFrame:
        """
        df = self._fetch_df_by_index(self._data, selector, level)
        df = self._fetch_df_by_col(df, col_set)
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
        trading_dates = self._data.index.unique(level='datetime')
        cur_loc = trading_dates.get_loc(cur_date)
        pre_loc = cur_loc - periods + 1
        if pre_loc < 0:
            warnings.warn('`periods` is too large. the first date will be returned.')
            pre_loc = 0
        ref_date = trading_dates[pre_loc]
        return slice(ref_date, cur_date)

    def get_range_iterator(self, periods: int, min_periods: Optional[int] = None,
                           **kwargs) -> Iterator[Tuple[pd.Timestamp, pd.DataFrame]]:
        """
        get a iterator of sliced data with given periods

        Args:
            periods (int): number of periods
            min_periods (int): minimum periods for sliced dataframe
            kwargs (dict): will be passed to `self.fetch`
        """
        trading_dates = self._data.index.unique(level='datetime')
        if min_periods is None:
            min_periods = periods
        for cur_date in trading_dates[min_periods:]:
            selector = self.get_range_selector(cur_date, periods)
            yield cur_date, self.fetch(selector, **kwargs)


class DataHandlerLP(DataHandler):
    """
    DataHandler with **(L)earnable (P)rocessor**
    """

    # data key
    DK_R = "raw"
    DK_I = "infer"
    DK_L = "learn"

    # process type
    PTYPE_I = "independent"
    # - _proc_infer_df will processed by infer_processors
    # - _proc_learn_df will be processed by learn_processors
    PTYPE_A = "append"
    # - _proc_infer_df will processed by infer_processors
    # - _proc_learn_df will be processed by infer_processors + learn_processors
    #   - (e.g. _proc_infer_df processed by learn_processors )

    def __init__(
        self,
        instruments,
        start_time=None,
        end_time=None,
        data_loader: Tuple[dict, str, DataLoader] = None,
        infer_processors=[],
        learn_processors=[],
        process_type=PTYPE_A,
        **kwargs,
    ):
        """
        Parameters
        ----------
        infer_processors : list
            list of <description info> of processors to generate data for inference
            example of <description info>:
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
            - _proc_infer_df will processed by infer_processors
            - _proc_learn_df will be processed by learn_processors
            PTYPE_A = 'append'
            - _proc_infer_df will processed by infer_processors
            - _proc_learn_df will be processed by infer_processors + learn_processors
              - (e.g. _proc_infer_df processed by learn_processors )
        """

        # Setup preprocessor
        self.infer_processors = []  # for lint
        self.learn_processors = []  # for lint
        for pname in "infer_processors", "learn_processors":
            for proc in locals()[pname]:
                getattr(self, pname).append(
                    init_instance_by_config(proc, processor_module, accept_types=(processor_module.Processor,))
                )

        self.process_type = process_type
        super().__init__(instruments, start_time, end_time, data_loader, **kwargs)

    def get_all_processors(self):
        return self.infer_processors + self.learn_processors

    def fit(self):
        for proc in self.get_all_processors():
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
        if len(self.infer_processors) > 0:  # avoid modifying the original  data
            _infer_df = _infer_df.copy()

        for proc in self.infer_processors:
            if not proc.is_for_infer():
                raise TypeError("Only processors usable for inference can be used in `infer_processors` ")
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
            if with_fit:
                proc.fit(_learn_df)
            _learn_df = proc(_learn_df)
        self._learn = _learn_df

    # init type
    IT_FIT_SEQ = "fit_seq"  # the input of `fit` will be the output of the previous processor
    IT_FIT_IND = "fit_ind"  # the input of `fit` will be the original df
    IT_LS = "load_state"  # The state of the object has been load by pickle

    def init(self, init_type: str = IT_FIT_SEQ, enable_cache: bool = False):
        """
        Initialize the data of Qlib

        Parameters
        ----------
        init_type : str
            The type `IT_*` listed above
        enable_cache : bool
            default value is false
            if `enable_cache` == True:
                the processed data will be saved on disk, and handler will load the cached data from the disk directly
                when we call `init` next time
        """
        # init raw data
        super().init(enable_cache=enable_cache)

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
        df = getattr(self, {self.DK_R: "_data", self.DK_I: "_infer", self.DK_L: "_learn"}[data_key])
        return df

    def fetch(
        self,
        selector: Union[pd.Timestamp, slice, str],
        level: Union[str, int] = "datetime",
        col_set=DataHandler.CS_ALL,
        data_key: str = DK_I,
    ) -> pd.DataFrame:
        """
        fetch data from underlying data source

        Parameters
        ----------
        selector : Union[pd.Timestamp, slice, str]
            describe how to select data by index
        level : Union[str, int]
            which index level to select the data
        col_set : str
            select a set of meaningful columns.(e.g. features, columns)
        data_key: str
            The data to fetch:  DK_*

        Returns
        -------
        pd.DataFrame:
        """
        df = self._get_df_by_key(data_key)
        df = self._fetch_df_by_index(df, selector, level)
        return self._fetch_df_by_col(df, col_set)

    def get_cols(self, col_set=DataHandler.CS_ALL, data_key: str = DK_I) -> list:
        """
        get the column names

        Parameters
        ----------
        col_set : str
            select a set of meaningful columns.(e.g. features, columns)
        data_key: str
            The data to fetch:  DK_*

        Returns
        -------
        list:
            list of column names
        """
        df = self._get_df_by_key(data_key).head()
        df = self._fetch_df_by_col(df, col_set)
        return df.columns.to_list()

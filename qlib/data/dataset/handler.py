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
from .utils import fetch_df_by_index, fetch_df_by_col
from ...utils import lazy_sort_index
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
            The stock list to retrieve.
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
            # make sure the fetch method is based on a index-sorted pd.DataFrame
            self._data = lazy_sort_index(self.data_loader.load(self.instruments, self.start_time, self.end_time))
        # TODO: cache

    CS_ALL = "__all"  # return all columns with single-level index column
    CS_RAW = "__raw"  # return raw data with multi-level index column

    def fetch(
        self,
        selector: Union[pd.Timestamp, slice, str, pd.Index] = slice(None, None),
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
            It can be categories as following
            - fetch single index
            - fetch a range of index
                - a slice range
                - pd.Index for specific indexes

            Following conflictions may occurs
            - Does [20200101", "20210101"] mean selecting this slice or these two days?
                - slice have higher priorities

        level : Union[str, int]
            which index level to select the data

        col_set : Union[str, List[str]]

            - if isinstance(col_set, str):

                select a set of meaningful, pd.Index columns.(e.g. features, columns)

                if col_set == CS_RAW:
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
        return self._fetch_data(
            data_storage=self._data,
            selector=selector,
            level=level,
            col_set=col_set,
            squeeze=squeeze,
            proc_func=proc_func,
        )

    def _fetch_data(
        self,
        data_storage,
        selector: Union[pd.Timestamp, slice, str, pd.Index] = slice(None, None),
        level: Union[str, int] = "datetime",
        col_set: Union[str, List[str]] = CS_ALL,
        squeeze: bool = False,
        proc_func: Callable = None,
    ):
        # This method is extracted for sharing in subclasses
        from .storage import BaseHandlerStorage

        # Following conflictions may occurs
        # - Does [20200101", "20210101"] mean selecting this slice or these two days?
        # To solve this issue
        #   - slice have higher priorities (except when level is none)
        if isinstance(selector, (tuple, list)) and level is not None:
            # when level is None, the argument will be passed in directly
            # we don't have to convert it into slice
            try:
                selector = slice(*selector)
            except ValueError:
                get_module_logger("DataHandlerLP").info(f"Fail to converting to query to slice. It will used directly")

        if isinstance(data_storage, pd.DataFrame):
            data_df = data_storage
            if proc_func is not None:
                # FIXME: fetching by time first will be more friendly to `proc_func`
                # Copy in case of `proc_func` changing the data inplace....
                data_df = proc_func(fetch_df_by_index(data_df, selector, level, fetch_orig=self.fetch_orig).copy())
                data_df = fetch_df_by_col(data_df, col_set)
            else:
                # Fetch column  first will be more friendly to SepDataFrame
                data_df = fetch_df_by_col(data_df, col_set)
                data_df = fetch_df_by_index(data_df, selector, level, fetch_orig=self.fetch_orig)
        elif isinstance(data_storage, BaseHandlerStorage):
            if not data_storage.is_proc_func_supported():
                if proc_func is not None:
                    raise ValueError(f"proc_func is not supported by the storage {type(data_storage)}")
                data_df = data_storage.fetch(
                    selector=selector, level=level, col_set=col_set, fetch_orig=self.fetch_orig
                )
            else:
                data_df = data_storage.fetch(
                    selector=selector, level=level, col_set=col_set, fetch_orig=self.fetch_orig, proc_func=proc_func
                )
        else:
            raise TypeError(f"data_storage should be pd.DataFrame|HasingStockStorage, not {type(data_storage)}")

        if squeeze:
            # squeeze columns
            data_df = data_df.squeeze()
            # squeeze index
            if isinstance(selector, (str, pd.Timestamp)):
                data_df = data_df.reset_index(level=level, drop=True)
        return data_df

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
        df = fetch_df_by_col(df, col_set)
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
    ATTR_MAP = {DK_R: "_data", DK_I: "_infer", DK_L: "_learn"}

    # process type
    PTYPE_I = "independent"
    # - self._infer will be processed by shared_processors + infer_processors
    # - self._learn will be processed by shared_processors + learn_processors

    # NOTE:
    PTYPE_A = "append"

    # - self._infer will be processed by shared_processors + infer_processors
    # - self._learn will be processed by shared_processors + infer_processors + learn_processors
    #   - (e.g. self._infer processed by learn_processors )

    def __init__(
        self,
        instruments=None,
        start_time=None,
        end_time=None,
        data_loader: Union[dict, str, DataLoader] = None,
        infer_processors: List = [],
        learn_processors: List = [],
        shared_processors: List = [],
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
        self.shared_processors = []  # for lint
        for pname in "infer_processors", "learn_processors", "shared_processors":
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
        return self.shared_processors + self.infer_processors + self.learn_processors

    def fit(self):
        """
        fit data without processing the data
        """
        for proc in self.get_all_processors():
            with TimeInspector.logt(f"{proc.__class__.__name__}"):
                proc.fit(self._data)

    def fit_process_data(self):
        """
        fit and process data

        The input of the `fit` will be the output of the previous processor
        """
        self.process_data(with_fit=True)

    @staticmethod
    def _run_proc_l(
        df: pd.DataFrame, proc_l: List[processor_module.Processor], with_fit: bool, check_for_infer: bool
    ) -> pd.DataFrame:
        for proc in proc_l:
            if check_for_infer and not proc.is_for_infer():
                raise TypeError("Only processors usable for inference can be used in `infer_processors` ")
            with TimeInspector.logt(f"{proc.__class__.__name__}"):
                if with_fit:
                    proc.fit(df)
                df = proc(df)
        return df

    @staticmethod
    def _is_proc_readonly(proc_l: List[processor_module.Processor]):
        """
        NOTE: it will return True if `len(proc_l) == 0`
        """
        for p in proc_l:
            if not p.readonly():
                return False
        return True

    def process_data(self, with_fit: bool = False):
        """
        process_data data. Fun `processor.fit` if necessary

        Notation: (data)  [processor]

        # data processing flow of self.process_type == DataHandlerLP.PTYPE_I
        (self._data)-[shared_processors]-(_shared_df)-[learn_processors]-(_learn_df)
                                               \
                                                -[infer_processors]-(_infer_df)

        # data processing flow of self.process_type == DataHandlerLP.PTYPE_A
        (self._data)-[shared_processors]-(_shared_df)-[infer_processors]-(_infer_df)-[learn_processors]-(_learn_df)

        Parameters
        ----------
        with_fit : bool
            The input of the `fit` will be the output of the previous processor
        """
        # shared data processors
        # 1) assign
        _shared_df = self._data
        if not self._is_proc_readonly(self.shared_processors):  # avoid modifying the original data
            _shared_df = _shared_df.copy()
        # 2) process
        _shared_df = self._run_proc_l(_shared_df, self.shared_processors, with_fit=with_fit, check_for_infer=True)

        # data for inference
        # 1) assign
        _infer_df = _shared_df
        if not self._is_proc_readonly(self.infer_processors):  # avoid modifying the original data
            _infer_df = _infer_df.copy()
        # 2) process
        _infer_df = self._run_proc_l(_infer_df, self.infer_processors, with_fit=with_fit, check_for_infer=True)

        self._infer = _infer_df

        # data for learning
        # 1) assign
        if self.process_type == DataHandlerLP.PTYPE_I:
            _learn_df = self._data
        elif self.process_type == DataHandlerLP.PTYPE_A:
            # based on `infer_df` and append the processor
            _learn_df = _infer_df
        else:
            raise NotImplementedError(f"This type of input is not supported")
        if not self._is_proc_readonly(self.learn_processors):  # avoid modifying the original  data
            _learn_df = _learn_df.copy()
        # 2) process
        _learn_df = self._run_proc_l(_learn_df, self.learn_processors, with_fit=with_fit, check_for_infer=False)

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
        df = getattr(self, self.ATTR_MAP[data_key])
        return df

    def fetch(
        self,
        selector: Union[pd.Timestamp, slice, str] = slice(None, None),
        level: Union[str, int] = "datetime",
        col_set=DataHandler.CS_ALL,
        data_key: str = DK_I,
        squeeze: bool = False,
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
        from .storage import BaseHandlerStorage

        return self._fetch_data(
            data_storage=self._get_df_by_key(data_key),
            selector=selector,
            level=level,
            col_set=col_set,
            squeeze=squeeze,
            proc_func=proc_func,
        )

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
        df = fetch_df_by_col(df, col_set)
        return df.columns.to_list()

    @classmethod
    def cast(cls, handler: "DataHandlerLP") -> "DataHandlerLP":
        """
        Motivation
        - A user create a datahandler in his customized package. Then he want to share the processed handler to other users without introduce the package dependency and complicated data processing logic.
        - This class make it possible by casting the class to DataHandlerLP and only keep the processed data

        Parameters
        ----------
        handler : DataHandlerLP
            A subclass of DataHandlerLP

        Returns
        -------
        DataHandlerLP:
            the converted processed data
        """
        new_hd: DataHandlerLP = object.__new__(DataHandlerLP)
        new_hd.from_cast = True  # add a mark for the casted instance

        for key in list(DataHandlerLP.ATTR_MAP.values()) + [
            "instruments",
            "start_time",
            "end_time",
            "fetch_orig",
            "drop_raw",
        ]:
            setattr(new_hd, key, getattr(handler, key, None))
        return new_hd

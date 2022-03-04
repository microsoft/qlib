# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import lru_cache
import logging
from typing import List, Text, Union, Callable, Iterable, Dict
from collections import OrderedDict

import inspect
import pandas as pd
import numpy as np

from ..utils.index_data import IndexData, SingleData
from ..utils.resam import resam_ts_data, ts_data_last
from ..log import get_module_logger
from ..utils.time import is_single_value, Freq
import qlib.utils.index_data as idd


class BaseQuote:
    def __init__(self, quote_df: pd.DataFrame, freq):
        self.logger = get_module_logger("online operator", level=logging.INFO)

    def get_all_stock(self) -> Iterable:
        """return all stock codes

        Return
        ------
        Iterable
            all stock codes
        """

        raise NotImplementedError(f"Please implement the `get_all_stock` method")

    def get_data(
        self,
        stock_id: str,
        start_time: Union[pd.Timestamp, str],
        end_time: Union[pd.Timestamp, str],
        field: Union[str],
        method: Union[str, None] = None,
    ) -> Union[None, int, float, bool, IndexData]:
        """get the specific field of stock data during start time and end_time,
           and apply method to the data.

           Example:
            .. code-block::
                                        $close      $volume
                instrument  datetime
                SH600000    2010-01-04  86.778313   16162960.0
                            2010-01-05  87.433578   28117442.0
                            2010-01-06  85.713585   23632884.0
                            2010-01-07  83.788803   20813402.0
                            2010-01-08  84.730675   16044853.0

                SH600655    2010-01-04  2699.567383  158193.328125
                            2010-01-08  2612.359619   77501.406250
                            2010-01-11  2712.982422  160852.390625
                            2010-01-12  2788.688232  164587.937500
                            2010-01-13  2790.604004  145460.453125

                this function is used for three case:

                1. method is not None. It returns int/float/bool/None.
                    - It will return None in one case, the method return None

                    print(get_data(stock_id="SH600000", start_time="2010-01-04", end_time="2010-01-06", field="$close", method="last"))

                    85.713585

                2. method is None. It returns IndexData.
                    print(get_data(stock_id="SH600000", start_time="2010-01-04", end_time="2010-01-06", field="$close", method=None))

                    IndexData([86.778313, 87.433578, 85.713585], [2010-01-04, 2010-01-05, 2010-01-06])

        Parameters
        ----------
        stock_id: str
        start_time : Union[pd.Timestamp, str]
            closed start time for backtest
        end_time : Union[pd.Timestamp, str]
            closed end time for backtest
        field : str
            the columns of data to fetch
        method : Union[str, None]
            the method apply to data.
            e.g [None, "last", "all", "sum", "mean", "ts_data_last"]

        Return
        ----------
        Union[None, int, float, bool, IndexData]
            it will return None in following cases
            - There is no stock data which meet the query criterion from data source.
            - The `method` returns None
        """

        raise NotImplementedError(f"Please implement the `get_data` method")


class PandasQuote(BaseQuote):
    def __init__(self, quote_df: pd.DataFrame, freq):
        super().__init__(quote_df=quote_df, freq=freq)
        quote_dict = {}
        for stock_id, stock_val in quote_df.groupby(level="instrument"):
            quote_dict[stock_id] = stock_val.droplevel(level="instrument")
        self.data = quote_dict

    def get_all_stock(self):
        return self.data.keys()

    def get_data(self, stock_id, start_time, end_time, field, method=None):
        if method == "ts_data_last":
            method = ts_data_last
        stock_data = resam_ts_data(self.data[stock_id][field], start_time, end_time, method=method)
        if stock_data is None:
            return None
        elif isinstance(stock_data, (bool, np.bool_, int, float, np.number)):
            return stock_data
        elif isinstance(stock_data, pd.Series):
            return idd.SingleData(stock_data)
        else:
            raise ValueError(f"stock data from resam_ts_data must be a number, pd.Series or pd.DataFrame")


class NumpyQuote(BaseQuote):
    def __init__(self, quote_df: pd.DataFrame, freq, region="cn"):
        """NumpyQuote

        Parameters
        ----------
        quote_df : pd.DataFrame
            the init dataframe from qlib.
        self.data : Dict(stock_id, IndexData.DataFrame)
        """
        super().__init__(quote_df=quote_df, freq=freq)
        quote_dict = {}
        for stock_id, stock_val in quote_df.groupby(level="instrument"):
            quote_dict[stock_id] = idd.MultiData(stock_val.droplevel(level="instrument"))
            quote_dict[stock_id].sort_index()  # To support more flexible slicing, we must sort data first
        self.data = quote_dict

        n, unit = Freq.parse(freq)
        if unit in Freq.SUPPORT_CAL_LIST:
            self.freq = Freq.get_timedelta(1, unit)
        else:
            raise ValueError(f"{freq} is not supported in NumpyQuote")
        self.region = region

    def get_all_stock(self):
        return self.data.keys()

    @lru_cache(maxsize=512)
    def get_data(self, stock_id, start_time, end_time, field, method=None):
        # check stock id
        if stock_id not in self.get_all_stock():
            return None

        # single data
        # If it don't consider the classification of single data, it will consume a lot of time.
        if is_single_value(start_time, end_time, self.freq, self.region):
            # this is a very special case.
            # skip aggregating function to speed-up the query calculation

            # FIXME:
            # it will go to the else logic when it comes to the
            # 1) the day before holiday when daily trading
            # 2) the last minute of the day when intraday trading
            try:
                return self.data[stock_id].loc[start_time, field]
            except KeyError:
                return None
        else:
            data = self.data[stock_id].loc[start_time:end_time, field]
            if data.empty:
                return None
            if method is not None:
                data = self._agg_data(data, method)
            return data

    def _agg_data(self, data: IndexData, method):
        """Agg data by specific method."""
        # FIXME: why not call the method of data directly?
        if method == "sum":
            return np.nansum(data)
        elif method == "mean":
            return np.nanmean(data)
        elif method == "last":
            # FIXME: I've never seen that this method was called.
            # Please merge it with "ts_data_last"
            return data[-1]
        elif method == "all":
            return data.all()
        elif method == "ts_data_last":
            valid_data = data.loc[~data.isna().data.astype(bool)]
            if len(valid_data) == 0:
                return None
            else:
                return valid_data.iloc[-1]
        else:
            raise ValueError(f"{method} is not supported")


class BaseSingleMetric:
    """
    The data structure of the single metric.
    The following methods are used for computing metrics in one indicator.
    """

    def __init__(self, metric: Union[dict, pd.Series]):
        """Single data structure for each metric.

        Parameters
        ----------
        metric : Union[dict, pd.Series]
            keys/index is stock_id, value is the metric value.
            for example:
                SH600068    NaN
                SH600079    1.0
                SH600266    NaN
                           ...
                SZ300692    NaN
                SZ300719    NaN,
        """
        raise NotImplementedError(f"Please implement the `__init__` method")

    def __add__(self, other: Union["BaseSingleMetric", int, float]) -> "BaseSingleMetric":
        raise NotImplementedError(f"Please implement the `__add__` method")

    def __radd__(self, other: Union["BaseSingleMetric", int, float]) -> "BaseSingleMetric":
        return self + other

    def __sub__(self, other: Union["BaseSingleMetric", int, float]) -> "BaseSingleMetric":
        raise NotImplementedError(f"Please implement the `__sub__` method")

    def __rsub__(self, other: Union["BaseSingleMetric", int, float]) -> "BaseSingleMetric":
        raise NotImplementedError(f"Please implement the `__rsub__` method")

    def __mul__(self, other: Union["BaseSingleMetric", int, float]) -> "BaseSingleMetric":
        raise NotImplementedError(f"Please implement the `__mul__` method")

    def __truediv__(self, other: Union["BaseSingleMetric", int, float]) -> "BaseSingleMetric":
        raise NotImplementedError(f"Please implement the `__truediv__` method")

    def __eq__(self, other: Union["BaseSingleMetric", int, float]) -> "BaseSingleMetric":
        raise NotImplementedError(f"Please implement the `__eq__` method")

    def __gt__(self, other: Union["BaseSingleMetric", int, float]) -> "BaseSingleMetric":
        raise NotImplementedError(f"Please implement the `__gt__` method")

    def __lt__(self, other: Union["BaseSingleMetric", int, float]) -> "BaseSingleMetric":
        raise NotImplementedError(f"Please implement the `__lt__` method")

    def __len__(self) -> int:
        raise NotImplementedError(f"Please implement the `__len__` method")

    def sum(self) -> float:
        raise NotImplementedError(f"Please implement the `sum` method")

    def mean(self) -> float:
        raise NotImplementedError(f"Please implement the `mean` method")

    def count(self) -> int:
        """Return the count of the single metric, NaN is not included."""

        raise NotImplementedError(f"Please implement the `count` method")

    def abs(self) -> "BaseSingleMetric":
        raise NotImplementedError(f"Please implement the `abs` method")

    @property
    def empty(self) -> bool:
        """If metric is empty, return True."""

        raise NotImplementedError(f"Please implement the `empty` method")

    def add(self, other: "BaseSingleMetric", fill_value: float = None) -> "BaseSingleMetric":
        """Replace np.NaN with fill_value in two metrics and add them."""

        raise NotImplementedError(f"Please implement the `add` method")

    def replace(self, replace_dict: dict) -> "BaseSingleMetric":
        """Replace the value of metric according to replace_dict."""

        raise NotImplementedError(f"Please implement the `replace` method")

    def apply(self, func: dict) -> "BaseSingleMetric":
        """Replace the value of metric with func(metric).
        Currently, the func is only qlib/backtest/order/Order.parse_dir.
        """

        raise NotImplementedError(f"Please implement the 'apply' method")


class BaseOrderIndicator:
    """
    The data structure of order indicator.
    !!!NOTE: There are two ways to organize the data structure. Please choose a better way.
        1. One way is using BaseSingleMetric to represent each metric. For example, the data
        structure of PandasOrderIndicator is Dict[str, PandasSingleMetric]. It uses
        PandasSingleMetric based on pd.Series to represent each metric.
        2. The another way doesn't use BaseSingleMetric to represent each metric. The data
        structure of PandasOrderIndicator is a whole matrix. It means you are not necessary
        to inherit the BaseSingleMetric.
    """

    def __init__(self, data):
        self.data = data
        self.logger = get_module_logger("online operator")

    def assign(self, col: str, metric: Union[dict, pd.Series]):
        """assign one metric.

        Parameters
        ----------
        col : str
            the metric name of one metric.
        metric : Union[dict, pd.Series]
            one metric with stock_id index, such as deal_amount, ffr, etc.
            for example:
                SH600068    NaN
                SH600079    1.0
                SH600266    NaN
                           ...
                SZ300692    NaN
                SZ300719    NaN,
        """

        raise NotImplementedError(f"Please implement the 'assign' method")

    def transfer(self, func: Callable, new_col: str = None) -> Union[None, BaseSingleMetric]:
        """compute new metric with existing metrics.

        Parameters
        ----------
        func : Callable
            the func of computing new metric.
            the kwargs of func will be replaced with metric data by name in this function.
            e.g.
                def func(pa):
                    return (pa > 0).sum() / pa.count()
        new_col : str, optional
            New metric will be assigned in the data if new_col is not None, by default None.

        Return
        ----------
        BaseSingleMetric
            new metric.
        """
        func_sig = inspect.signature(func).parameters.keys()
        func_kwargs = {sig: self.data[sig] for sig in func_sig}
        tmp_metric = func(**func_kwargs)
        if new_col is not None:
            self.data[new_col] = tmp_metric
        else:
            return tmp_metric

    def get_metric_series(self, metric: str) -> pd.Series:
        """return the single metric with pd.Series format.

        Parameters
        ----------
        metric : str
            the metric name.

        Return
        ----------
        pd.Series
            the single metric.
            If there is no metric name in the data, return pd.Series().
        """

        raise NotImplementedError(f"Please implement the 'get_metric_series' method")

    def get_index_data(self, metric) -> SingleData:
        """get one metric with the format of SingleData

        Parameters
        ----------
        metric : str
            the metric name.

        Return
        ------
        IndexData.Series
            one metric with the format of SingleData
        """

        raise NotImplementedError(f"Please implement the 'get_index_data' method")

    @staticmethod
    def sum_all_indicators(order_indicator, indicators: list, metrics: Union[str, List[str]], fill_value: float = None):
        """sum indicators with the same metrics.
        and assign to the order_indicator(BaseOrderIndicator).
        NOTE: indicators could be a empty list when orders in lower level all fail.

        Parameters
        ----------
        order_indicator : BaseOrderIndicator
            the order indicator to assign.
        indicators : List[BaseOrderIndicator]
            the list of all inner indicators.
        metrics : Union[str, List[str]]
            all metrics needs to be sumed.
        fill_value : float, optional
            fill np.NaN with value. By default None.
        """

        raise NotImplementedError(f"Please implement the 'sum_all_indicators' method")

    def to_series(self) -> Dict[Text, pd.Series]:
        """return the metrics as pandas series

        for example: { "ffr":
                SH600068    NaN
                SH600079    1.0
                SH600266    NaN
                           ...
                SZ300692    NaN
                SZ300719    NaN,
                ...
         }
        """
        raise NotImplementedError(f"Please implement the `to_series` method")


class SingleMetric(BaseSingleMetric):
    def __init__(self, metric):
        self.metric = metric

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(self.metric + other)
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric + other.metric)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(self.metric - other)
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric - other.metric)
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(other - self.metric)
        elif isinstance(other, self.__class__):
            return self.__class__(other.metric - self.metric)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(self.metric * other)
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric * other.metric)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(self.metric / other)
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric / other.metric)
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(self.metric == other)
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric == other.metric)
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(self.metric > other)
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric > other.metric)
        else:
            return NotImplemented

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(self.metric < other)
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric < other.metric)
        else:
            return NotImplemented

    def __len__(self):
        return len(self.metric)


class PandasSingleMetric(SingleMetric):
    """Each SingleMetric is based on pd.Series."""

    def __init__(self, metric: Union[dict, pd.Series] = {}):
        if isinstance(metric, dict):
            self.metric = pd.Series(metric)
        elif isinstance(metric, pd.Series):
            self.metric = metric
        else:
            raise ValueError(f"metric must be dict or pd.Series")

    def sum(self):
        return self.metric.sum()

    def mean(self):
        return self.metric.mean()

    def count(self):
        return self.metric.count()

    def abs(self):
        return self.__class__(self.metric.abs())

    @property
    def empty(self):
        return self.metric.empty

    @property
    def index(self):
        return list(self.metric.index)

    def add(self, other, fill_value=None):
        return self.__class__(self.metric.add(other.metric, fill_value=fill_value))

    def replace(self, replace_dict: dict):
        return self.__class__(self.metric.replace(replace_dict))

    def apply(self, func: Callable):
        return self.__class__(self.metric.apply(func))

    def reindex(self, index, fill_value):
        return self.__class__(self.metric.reindex(index, fill_value=fill_value))

    def __repr__(self):
        return repr(self.metric)


class PandasOrderIndicator(BaseOrderIndicator):
    """
    The data structure is OrderedDict(str: PandasSingleMetric).
    Each PandasSingleMetric based on pd.Series is one metric.
    Str is the name of metric.
    """

    def __init__(self):
        self.data: Dict[str, PandasSingleMetric] = OrderedDict()

    def assign(self, col: str, metric: Union[dict, pd.Series]):
        self.data[col] = PandasSingleMetric(metric)

    def get_index_data(self, metric):
        if metric in self.data:
            return idd.SingleData(self.data[metric].metric)
        else:
            return idd.SingleData()

    def get_metric_series(self, metric: str) -> Union[pd.Series]:
        if metric in self.data:
            return self.data[metric].metric
        else:
            return pd.Series()

    def to_series(self):
        return {k: v.metric for k, v in self.data.items()}

    @staticmethod
    def sum_all_indicators(order_indicator, indicators: list, metrics: Union[str, List[str]], fill_value=0):
        if isinstance(metrics, str):
            metrics = [metrics]
        for metric in metrics:
            tmp_metric = PandasSingleMetric({})
            for indicator in indicators:
                tmp_metric = tmp_metric.add(indicator.data[metric], fill_value)
            order_indicator.assign(metric, tmp_metric.metric)

    def __repr__(self):
        return repr(self.data)


class NumpyOrderIndicator(BaseOrderIndicator):
    """
    The data structure is OrderedDict(str: SingleData).
    Each idd.SingleData is one metric.
    Str is the name of metric.
    """

    def __init__(self):
        self.data: Dict[str, SingleData] = OrderedDict()

    def assign(self, col: str, metric: dict):
        self.data[col] = idd.SingleData(metric)

    def get_index_data(self, metric):
        if metric in self.data:
            return self.data[metric]
        else:
            return idd.SingleData()

    def get_metric_series(self, metric: str) -> Union[pd.Series]:
        return self.data[metric].to_series()

    def to_series(self) -> Dict[str, pd.Series]:
        tmp_metric_dict = {}
        for metric in self.data:
            tmp_metric_dict[metric] = self.get_metric_series(metric)
        return tmp_metric_dict

    @staticmethod
    def sum_all_indicators(order_indicator, indicators: list, metrics: Union[str, List[str]], fill_value=0):
        # get all index(stock_id)
        stocks = set()
        for indicator in indicators:
            # set(np.ndarray.tolist()) is faster than set(np.ndarray)
            stocks = stocks | set(indicator.data[metrics[0]].index.tolist())
        stocks = list(stocks)
        stocks.sort()

        # add metric by index
        if isinstance(metrics, str):
            metrics = [metrics]
        for metric in metrics:
            order_indicator.data[metric] = idd.sum_by_index(
                [indicator.data[metric] for indicator in indicators], stocks, fill_value
            )

    def __repr__(self):
        return repr(self.data)

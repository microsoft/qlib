# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import logging
from pandas._config.config import is_instance_factory
from qlib.data.base import Feature
from typing import List, Text, Tuple, Union, Callable, Iterable, Dict, ValuesView
from collections import OrderedDict

import inspect
import bisect
import pandas as pd
import numpy as np

from ..utils.resam import resam_ts_data, ts_data_last
from ..log import get_module_logger


class BaseQuote:
    def __init__(self, quote_df: pd.DataFrame):
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
        stock_id: Union[str, list],
        start_time: Union[pd.Timestamp, str],
        end_time: Union[pd.Timestamp, str],
        fields: Union[str, list] = None,
        method: Union[str, Callable] = None,
    ) -> Union[None, float, pd.Series, pd.DataFrame]:
        """get the specific fields of stock data during start time and end_time,
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

                print(get_data(stock_id=["SH600000", "SH600655"], start_time="2010-01-04", end_time="2010-01-05", fields=["$close", "$volume"], method="last"))

                            $close      $volume
                instrument
                SH600000    87.433578 28117442.0
                SH600655    2699.567383  158193.328125

                print(get_data(stock_id="SH600000", start_time="2010-01-04", end_time="2010-01-05", fields=["$close", "$volume"], method="last"))

                $close 87.433578
                $volume 28117442.0

                print(get_data(stock_id="SH600000", start_time="2010-01-04", end_time="2010-01-05", fields="$close", method="last"))

                87.433578

        Parameters
        ----------
        stock_id: Union[str, list]
        start_time : Union[pd.Timestamp, str]
            closed start time for backtest
        end_time : Union[pd.Timestamp, str]
            closed end time for backtest
        fields : Union[str, List]
            the columns of data to fetch
        method : Union[str, Callable]
            the method apply to data.
            e.g [None, "last", "all", "sum", "mean", "any", qlib/utils/resam.py/ts_data_last]

        Return
        ----------
        Union[None, float, pd.Series, pd.DataFrame]
            The resampled DataFrame/Series/value, return None when the resampled data is empty.
        """

        raise NotImplementedError(f"Please implement the `get_data` method")


class PandasQuote(BaseQuote):
    def __init__(self, quote_df: pd.DataFrame):
        super().__init__(quote_df=quote_df)
        quote_dict = {}
        for stock_id, stock_val in quote_df.groupby(level="instrument"):
            quote_dict[stock_id] = stock_val.droplevel(level="instrument")
        self.data = quote_dict

    def get_all_stock(self):
        return self.data.keys()

    def get_data(self, stock_id, start_time, end_time, fields=None, method=None):
        if fields is None:
            return resam_ts_data(self.data[stock_id], start_time, end_time, method=method)
        elif isinstance(fields, (str, list)):
            return resam_ts_data(self.data[stock_id][fields], start_time, end_time, method=method)
        else:
            raise ValueError(f"fields must be None, str or list")

    def _if_single_data(self, start_time, end_time):
        if end_time - start_time < np.timedelta64(1, "m"):
            return True
        if start_time.hour == 11 and start_time.minute == 29 and start_time.second == 0:
            return True
        if start_time.hour == 14 and start_time.minute == 59 and start_time.second == 0:
            return True
        return False


class NumpyQuote(BaseQuote):
    def __init__(self, quote_df: pd.DataFrame):
        """NumpyQuote

        Parameters
        ----------
        quote_df : pd.DataFrame
            the init dataframe from qlib.

        Variables
        self.data: Dict[stock_id, np.ndarray]
            each stock has one two-dimensional np.ndarray to represent data.
        self.columns: Dict[str, int]
            map column name to column id in self.data.
        self.dates: Dict[stock_id, Dict[pd.Timestap, int]]
            map timestap to row id in self.data.
        self.dates_list: Dict[stock_id, List[pd.Timestap]]
            the dates of each stock for searching.
        """

        super().__init__(quote_df=quote_df)
        # init data
        columns = quote_df.columns.values
        self.columns = dict(zip(columns, range(len(columns))))
        self.data, self.dates, self.dates_list = self._to_numpy(quote_df)

        # lru
        self.muti_lru = {}
        self.max_lru_len = 256

    def _to_numpy(self, quote_df):
        """convert dataframe to numpy."""

        quote_dict = {}
        date_dict = {}
        date_list = {}
        for stock_id, stock_val in quote_df.groupby(level="instrument"):
            quote_dict[stock_id] = stock_val.values
            date_dict[stock_id] = stock_val.index.get_level_values("datetime")
            date_list[stock_id] = list(date_dict[stock_id])
        for stock_id in date_dict:
            date_dict[stock_id] = dict(zip(date_dict[stock_id], range(len(date_dict[stock_id]))))
        return quote_dict, date_dict, date_list

    def get_all_stock(self):
        return self.data.keys()

    def get_data(self, stock_id, start_time, end_time, fields=None, method=None):
        # check stock id
        if stock_id not in self.get_all_stock():
            return None

        # get single data
        if self._if_single_data(start_time, end_time):
            if start_time not in self.dates[stock_id]:
                return None
            if fields is None:
                # it used for check if data is None
                return self.data[stock_id][self.dates[stock_id][start_time]]
            else:
                return self.data[stock_id][self.dates[stock_id][start_time]][self.columns[fields]]
        # get muti row data
        else:
            # check lru
            if (stock_id, start_time, end_time, fields, method) in self.muti_lru:
                return self.muti_lru[(stock_id, start_time, end_time, fields, method)]

            start_id = bisect.bisect_left(self.dates_list[stock_id], start_time)
            end_id = bisect.bisect_right(self.dates_list[stock_id], end_time)
            if start_id == end_id:
                return None
            # it used for check if data is None
            if fields is None:
                return self.data[stock_id][start_id:end_id]
            agg_stock_data = self._agg_data(self.data[stock_id][start_id:end_id, self.columns[fields]], method)

            # result lru
            if len(self.muti_lru) >= self.max_lru_len:
                self.muti_lru.clear()
            self.muti_lru[(stock_id, start_time, end_time, fields, method)] = agg_stock_data
            return agg_stock_data

    def _agg_data(self, data, method):
        """Agg data by specific method."""

        if method == "sum":
            return data.sum()
        if method == "mean":
            return data.mean()
        if method == "last":
            return data[-1]
        if method == "all":
            return data.all()
        if method == "any":
            return data.any()
        if method == ts_data_last:
            valid_data = data[data != np.NaN]
            if len(valid_data) == 0:
                return None
            else:
                return valid_data[0]

    def _if_single_data(self, start_time, end_time):
        """Is there only one piece of data to obtaine.

        Parameters
        ----------
        start_time : Union[pd.Timestamp, str]
            closed start time for data.
        end_time : Union[pd.Timestamp, str]
            closed end time for data.
        Returns
        -------
        bool
            True means one piece of data to obtaine.
        """

        if end_time - start_time < np.timedelta64(1, "m"):
            return True
        if start_time.hour == 11 and start_time.minute == 29 and start_time.second == 0:
            return True
        if start_time.hour == 14 and start_time.minute == 59 and start_time.second == 0:
            return True
        return False


class BaseSingleMetric:
    """
    The data structure of the single metric.
    The following methods are used for computing metrics in one indicator.
    """

    def __init__(self, metric: Union[dict, pd.Series]):
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

    def astype(self, type: type) -> "BaseSingleMetric":
        raise NotImplementedError(f"Please implement the `astype` method")

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
        structure of PandasOrderIndicator is a whole matrix. It means you are not neccesary
        to inherit the BaseSingleMetric.
    """

    def assign(self, col: str, metric: Union[dict, pd.Series]):
        """assign one metric.

        Parameters
        ----------
        col : str
            the metric name of one metric.
        metric : Union[dict, pd.Series]
            the metric data.
        """

        pass

    def transfer(self, func: Callable, new_col: str = None) -> Union[None, BaseSingleMetric]:
        """compute new metric with existing metrics.

        Parameters
        ----------
        func : Callable
            the func of computing new metric.
            the kwargs of func will be replaced with metric data by name in this function.
            e.g.
                def func(pa):
                    return (pa > 0).astype(int).sum() / pa.count()
        new_col : str, optional
            New metric will be assigned in the data if new_col is not None, by default None.

        Return
        ----------
        BaseSingleMetric
            new metric.
        """

        pass

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

        pass

    @staticmethod
    def sum_all_indicators(
        cls, indicators: list, metrics: Union[str, List[str]], fill_value: float = None
    ):
        """sum indicators with the same metrics.
        and assign to the cls(BaseOrderIndicator).

        Parameters
        ----------
        cls : BaseOrderIndicator
            the order indicator to assign.
        indicators : List[BaseOrderIndicator]
            the list of all inner indicators.
        metrics : Union[str, List[str]]
            all metrics needs ot be sumed.
        fill_value : float, optional
            fill np.NaN with value. By default None.
        """

        pass

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


class PandasSingleMetric:
    """Each SingleMetric is based on pd.Series."""

    def __init__(self, metric: Union[dict, pd.Series]):
        if isinstance(metric, dict):
            self.metric = pd.Series(metric)
        elif isinstance(metric, pd.Series):
            self.metric = metric
        else:
            raise ValueError(f"metric must be dict or pd.Series")

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return PandasSingleMetric(self.metric + other)
        elif isinstance(other, PandasSingleMetric):
            return PandasSingleMetric(self.metric + other.metric)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return PandasSingleMetric(self.metric - other)
        elif isinstance(other, PandasSingleMetric):
            return PandasSingleMetric(self.metric - other.metric)
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return PandasSingleMetric(other - self.metric)
        elif isinstance(other, PandasSingleMetric):
            return PandasSingleMetric(other.metric - self.metric)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return PandasSingleMetric(self.metric * other)
        elif isinstance(other, PandasSingleMetric):
            return PandasSingleMetric(self.metric * other.metric)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return PandasSingleMetric(self.metric / other)
        elif isinstance(other, PandasSingleMetric):
            return PandasSingleMetric(self.metric / other.metric)
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return PandasSingleMetric(self.metric == other)
        elif isinstance(other, PandasSingleMetric):
            return PandasSingleMetric(self.metric == other.metric)
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return PandasSingleMetric(self.metric > other)
        elif isinstance(other, PandasSingleMetric):
            return PandasSingleMetric(self.metric > other.metric)
        else:
            return NotImplemented

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return PandasSingleMetric(self.metric < other)
        elif isinstance(other, PandasSingleMetric):
            return PandasSingleMetric(self.metric < other.metric)
        else:
            return NotImplemented

    def __len__(self):
        return len(self.metric)

    def sum(self):
        return self.metric.sum()

    def mean(self):
        return self.metric.mean()

    def count(self):
        return self.metric.count()

    def abs(self):
        return PandasSingleMetric(self.metric.abs())

    def astype(self, type):
        return PandasSingleMetric(self.metric.astype(type))

    @property
    def empty(self):
        return self.metric.empty

    def add(self, other, fill_value=None):
        return PandasSingleMetric(self.metric.add(other.metric, fill_value=fill_value))

    def replace(self, replace_dict: dict):
        return PandasSingleMetric(self.metric.replace(replace_dict))

    def apply(self, func: Callable):
        return PandasSingleMetric(self.metric.apply(func))


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

    def transfer(self, func: Callable, new_col: str = None) -> Union[None, PandasSingleMetric]:
        func_sig = inspect.signature(func).parameters.keys()
        func_kwargs = {sig: self.data[sig] for sig in func_sig}
        tmp_metric = func(**func_kwargs)
        if new_col is not None:
            self.data[new_col] = tmp_metric
        else:
            return tmp_metric

    def get_metric_series(self, metric: str) -> Union[pd.Series]:
        if metric in self.data:
            return self.data[metric].metric
        else:
            return pd.Series()

    @staticmethod
    def sum_all_indicators(
        cls, indicators: list, metrics: Union[str, List[str]], fill_value=None
    ):
        if isinstance(metrics, str):
            metrics = [metrics]
        for metric in metrics:
            tmp_metric = PandasSingleMetric({})
            for indicator in indicators:
                tmp_metric = tmp_metric.add(indicator.data[metric], fill_value)
            cls.assign(metric, tmp_metric.metric)

    def to_series(self):
        return {k: v.metric for k, v in self.data.items()}


class NumpySingleMetric(BaseSingleMetric):
    def __init__(self, metric: np.ndarray):
        self.metric = metric

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return NumpySingleMetric(self.metric + other)
        elif isinstance(other, NumpySingleMetric):
            return NumpySingleMetric(self.metric + other.metric)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return NumpySingleMetric(self.metric - other)
        elif isinstance(other, NumpySingleMetric):
            return NumpySingleMetric(self.metric - other.metric)
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return NumpySingleMetric(other - self.metric)
        elif isinstance(other, NumpySingleMetric):
            return NumpySingleMetric(other.metric - self.metric)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return NumpySingleMetric(self.metric * other)
        elif isinstance(other, NumpySingleMetric):
            return NumpySingleMetric(self.metric * other.metric)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return NumpySingleMetric(self.metric / other)
        elif isinstance(other, NumpySingleMetric):
            return NumpySingleMetric(self.metric / other.metric)
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return NumpySingleMetric(self.metric == other)
        elif isinstance(other, NumpySingleMetric):
            return NumpySingleMetric(self.metric == other.metric)
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return NumpySingleMetric(self.metric > other)
        elif isinstance(other, NumpySingleMetric):
            return NumpySingleMetric(self.metric > other.metric)
        else:
            return NotImplemented

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return NumpySingleMetric(self.metric < other)
        elif isinstance(other, NumpySingleMetric):
            return NumpySingleMetric(self.metric < other.metric)
        else:
            return NotImplemented

    def __len__(self):
        return len(self.metric)

    def sum(self):
        return self.metric.sum()

    def mean(self):
        return self.metric.mean()

    def count(self):
        return len(self.metric[~np.isnan(self.metric)])

    def abs(self):
        return NumpySingleMetric(np.absolute(self.metric))

    def astype(self, type):
        return NumpySingleMetric(self.metric.astype(type))

    @property
    def empty(self):
        return len(self.metric) == 0

    def replace(self, replace_dict: dict):
        tmp_metric = self.metric.copy()
        for num in replace_dict:
            tmp_metric[tmp_metric == num] = replace_dict[num]
        return NumpySingleMetric(tmp_metric)

    def apply(self, func: Callable):
        tmp_metric = self.metric.copy()
        for i in range(len(tmp_metric)):
            tmp_metric[i] = func(tmp_metric[i])
        return NumpySingleMetric(tmp_metric)


class NumpyOrderIndicator(BaseOrderIndicator):
    # all metrics
    ROW = [
        "amount",
        "deal_amount",
        "inner_amount",
        "trade_price",
        "trade_value",
        "trade_cost",
        "trade_dir",
        "ffr",
        "pa",
        "pos",
        "base_price",
        "base_volume",
    ]
    ROW_MAP = dict(zip(ROW, range(len(ROW))))

    def __init__(self):
        self.row_tag = [0 for tag in range(len(NumpyOrderIndicator.ROW))]
        self.data = None

    def assign(self, col: str, metric: Union[dict, np.ndarray, pd.Series]):
        if col not in NumpyOrderIndicator.ROW:
            raise ValueError(f"{col} metric is not supoorted")
        if not isinstance(metric, (dict, np.ndarray, pd.Series)):
            raise ValueError(f"metric must be dict, pd.Series or np.ndarray")
        if isinstance(metric, (pd.Series, np.ndarray)) and self.data is None:
            raise ValueError(f"data can not be None when metric is np.ndarray or pd.Series")

        # if data is None, init numpy ndarray
        if self.data is None:
            self.data = np.zeros((len(NumpyOrderIndicator.ROW), len(metric)))
            self.column = list(metric.keys())
            self.column_map = dict(zip(self.column, range(len(self.column))))
        
        metric_column = list(metric.keys())
        if self.column != metric_column:
            assert len(set(self.column) - set(metric_column)) == 0
            # modify the order
            tmp_metric = {}
            for column in self.column:
                tmp_metric[column] = metric[column]
            metric = tmp_metric

        # assign data
        self.row_tag[NumpyOrderIndicator.ROW_MAP[col]] = 1
        if isinstance(metric, dict):
            self.data[NumpyOrderIndicator.ROW_MAP[col]] = list(metric.values())
        elif isinstance(metric, np.ndarray):
            self.data[NumpyOrderIndicator.ROW_MAP[col]] = metric
        elif isinstance(metric, pd.Series):
            self.data[NumpyOrderIndicator.ROW_MAP[col]] = metric.values

    def transfer(self, func: Callable, new_col: str = None) -> Union[None, NumpySingleMetric]:
        func_sig = inspect.signature(func).parameters.keys()
        func_kwargs = {}
        for sig in func_sig:
            if self._if_valid_metric(sig):
                func_kwargs[sig] = NumpySingleMetric(self.data[NumpyOrderIndicator.ROW_MAP[sig]])
            else:
                print(f"{sig} is not assigned")
                func_kwargs[sig] = NumpySingleMetric(np.array([]))
        tmp_metric = func(**func_kwargs)
        if new_col is not None:
            self.row_tag[NumpyOrderIndicator.ROW_MAP[new_col]] = 1
            self.data[NumpyOrderIndicator.ROW_MAP[new_col]] = tmp_metric.metric
        else:
            return tmp_metric

    def get_metric_series(self, metric: str) -> Union[pd.Series]:
        if self._if_valid_metric(metric):
            return pd.Series(self.data[NumpyOrderIndicator.ROW_MAP[metric]], index=self.column)
        else:
            return pd.Series()

    def to_series(self) -> Dict[str, pd.Series]:
        tmp_metric_dict = {}
        for metric in NumpyOrderIndicator.ROW:
            tmp_metric_dict[metric] = self.get_metric_series(metric)
        return tmp_metric_dict

    def _if_valid_metric(self, metric):
        if metric in NumpyOrderIndicator.ROW and self.row_tag[NumpyOrderIndicator.ROW_MAP[metric]] == 1:
            return True
        else:
            return False

    @staticmethod
    def sum_all_indicators(
        cls, indicators: list, metrics: Union[str, List[str]], fill_value=None
    ) -> Dict[str, NumpySingleMetric]:
        # metrics is all metrics to add
        # metrics_id means the index in the NumpyOrderIndicator.ROW for metrics.
        if isinstance(metrics, str):
            metrics = [metrics]
        metrics_id = [NumpyOrderIndicator.ROW_MAP[metric] for metric in metrics]

        # get all stock_id and all metric data
        stocks = set()
        indicator_metrics = []
        for indicator in indicators:
            stocks = stocks | set(indicator.column)
            indicator_metrics.append(indicator.data[metrics_id, :].copy())
        stocks = list(stocks)
        stocks_map = dict(zip(stocks, range(len(stocks))))

        # fill value
        if fill_value is not None:
            base_metrics = fill_value * np.ones((len(metrics), len(stocks)))
            for i in range(len(indicators)):
                tmp_netrics = base_metrics.copy()
                stocks_index = [stocks_map[stock] for stock in indicators[i].column]
                tmp_netrics[:, stocks_index] = indicator_metrics[i]
                indicator_metrics[i] = tmp_netrics
        else:
            raise ValueError(f"fill value can not be None in NumpyOrderIndicator")

        # add metric and assign to cls 
        metric_sum = sum(indicator_metrics)
        if cls.data is not None:
            raise ValueError(f"this function must assign to an empty order indicator")
        cls.data = np.zeros((len(NumpyOrderIndicator.ROW), len(stocks)))
        cls.column = stocks
        cls.column_map = dict(zip(stocks, range(len(stocks))))
        for i in range(len(metrics)):
            cls.row_tag[NumpyOrderIndicator.ROW_MAP[metrics[i]]] = 1
            cls.data[NumpyOrderIndicator.ROW_MAP[metrics[i]]] = metric_sum[i]


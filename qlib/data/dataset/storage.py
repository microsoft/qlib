import pandas as pd
import numpy as np

from .handler import DataHandler
from typing import Tuple, Union, List, Callable

from .utils import get_level_index, fetch_df_by_index, fetch_df_by_col


class BaseHandlerStorage:
    def fetch(
        self,
        selector: Union[pd.Timestamp, slice, str, list] = slice(None, None),
        level: Union[str, int] = "datetime",
        col_set: Union[str, List[str]] = DataHandler.CS_ALL,
        fetch_orig: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """fetch data from the data storage

        Parameters
        ----------
        selector : Union[pd.Timestamp, slice, str]
            describe how to select data by index
        level : Union[str, int]
            which index level to select the data
        col_set : Union[str, List[str]]
            - if isinstance(col_set, str):
                select a set of meaningful columns.(e.g. features, columns)
                if col_set == DataHandler.CS_RAW:
                    the raw dataset will be returned.
            - if isinstance(col_set, List[str]):
                select several sets of meaningful columns, the returned data has multiple level
        fetch_orig : bool
            Return the original data instead of copy if possible.

        """

        raise NotImplementedError("fetch is method not implemented!")

    @staticmethod
    def from_df(df: pd.DataFrame):
        raise NotImplementedError("from_df method is not implemented!")


class HasingStockStorage(BaseHandlerStorage):
    def __init__(self, df):
        self.hash_df = dict()
        self.stock_level = get_level_index(df, "instrument")
        for k, v in df.groupby(level="instrument"):
            self.hash_df[k] = v
        self.columns = df.columns

    @staticmethod
    def from_df(df):
        return HasingStockStorage(df)

    def _fetch_hash_df_by_stock(self, selector, level):
        stock_selector = slice(None)

        if level is None:
            if isinstance(selector, tuple) and self.stock_level < len(selector):
                stock_selector = selector[self.stock_level]
            elif isinstance(selector, (list, str)) and self.stock_level == 0:
                stock_selector = selector
        elif level == "instrument" or level == self.stock_level:
            if isinstance(selector, tuple):
                stock_selector = selector[0]
            elif isinstance(selector, (list, str)):
                stock_selector = selector

        if not isinstance(stock_selector, (list, str)) and stock_selector != slice(None):
            raise TypeError(f"stock selector must be type str|list, or slice(None), rather than {stock_selector}")
        print(stock_selector)
        if stock_selector == slice(None):
            return self.hash_df

        if isinstance(stock_selector, str):
            stock_selector = [stock_selector]

        select_dict = dict()
        for each_stock in sorted(stock_selector):
            if each_stock in self.hash_df:
                select_dict[each_stock] = self.hash_df[each_stock]
        return select_dict

    def fetch(
        self,
        selector: Union[pd.Timestamp, slice, str] = slice(None, None),
        level: Union[str, int] = "datetime",
        col_set: Union[str, List[str]] = DataHandler.CS_ALL,
        fetch_orig: bool = True,
    ) -> pd.DataFrame:
        fetch_stock_df_list = list(self._fetch_hash_df_by_stock(selector=selector, level=level).values())
        for _index, stock_df in enumerate(fetch_stock_df_list):
            fetch_col_df = fetch_df_by_col(df=stock_df, col_set=col_set)
            fetch_index_df = fetch_df_by_index(df=fetch_col_df, selector=selector, level=level, fetch_orig=fetch_orig)
            fetch_stock_df_list[_index] = fetch_index_df
        if len(fetch_stock_df_list) == 0:
            index_names = ("instrument", "datetime") if self.stock_level == 0 else ("datetime", "instrument")
            return pd.DataFrame(
                index=pd.MultiIndex.from_arrays([[], []], names=index_names), columns=self.columns, dtype=np.float32
            )
        elif len(fetch_stock_df_list) == 1:
            return fetch_stock_df_list[0]
        else:
            return pd.concat(fetch_stock_df_list, sort=False, copy=~fetch_orig)

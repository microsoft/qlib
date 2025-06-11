from abc import ABC, abstractmethod
import pandas as pd
from typing import Union, List

class DataContainer(ABC):
    """所有数据容器的抽象基类。"""
    @abstractmethod
    def get_data(self):
        """返回底层的数据结构。"""
        pass

    @abstractmethod
    def get_shape(self) -> tuple:
        """返回数据的形状。"""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.get_shape()})"

class PanelContainer(DataContainer):
    """
    用于存储面板数据 (Time x Stock) 的容器, 使用 pd.DataFrame 实现。
    index 为 datetime, columns 为 stock_id。
    """
    def __init__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("传入的数据必须是 pandas.DataFrame。")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("DataFrame 的索引必须是 pd.DatetimeIndex。")
        self._data = data

    def get_data(self) -> pd.DataFrame:
        """返回底层的 DataFrame。"""
        return self._data

    def get_shape(self) -> tuple:
        """返回数据矩阵的形状。"""
        return self._data.shape

    def reindex(self, dates: pd.DatetimeIndex, stocks: List[str]) -> 'PanelContainer':
        """
        将容器的数据重新索引到一组新的日期和股票。
        """
        reindexed_data = self._data.reindex(index=dates, columns=stocks)
        return PanelContainer(reindexed_data)

    def __repr__(self):
        return f"PanelContainer(shape={self.get_shape()})\n{self._data.head()}" 

    def get_dates(self) -> pd.DatetimeIndex:
        """返回数据矩阵的日期索引。"""
        return self._data.index

    def get_stocks(self) -> List[str]:
        """返回数据矩阵的股票列名。"""
        return self._data.columns.tolist()
    
    def get_stock_data(self, stock: str) -> pd.Series:
        """返回指定股票的时间序列数据。"""
        return self._data[stock]
    
    def get_date_data(self, date: pd.Timestamp) -> pd.Series:
        """返回指定日期的截面数据。"""
        return self._data.loc[date]

class CrossSectionContainer(DataContainer):
    """
    用于存储截面数据 (1 x Stock) 的容器, 使用 pd.Series 实现。
    """
    def __init__(self, data: pd.Series):
        if not isinstance(data, pd.Series):
            raise TypeError("传入的数据必须是 pandas.Series。")
        self._data = data

    def get_data(self) -> pd.Series:
        """返回底层的 Series。"""
        return self._data

    def get_shape(self) -> tuple:
        """返回数据序列的形状。"""
        return self._data.shape

    def get_stocks(self) -> List[str]:
        """返回数据序列的股票索引。"""
        return self._data.index.tolist()
    
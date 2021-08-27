# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from typing import Union, Callable
import bisect

import numpy as np
import pandas as pd


class IndexData:
    """This is a simplified version of pandas which is faster based on numpy."""

    @staticmethod
    def Series(
        data: Union[dict, pd.Series, int, float, np.floating, list, np.ndarray] = [], index: Union[list, pd.Index] = []
    ):
        if isinstance(data, dict):
            return SingleData(list(data.values()), list(data.keys()))
        elif isinstance(data, pd.Series):
            return SingleData(data.values, data.index)
        else:
            return SingleData(data, index)

    @staticmethod
    def DataFrame(
        data: Union[pd.DataFrame, list, np.ndarray] = [[]],
        index: Union[list, pd.Index] = [],
        columns: Union[list, pd.Index] = [],
    ):
        if isinstance(data, pd.DataFrame):
            return MultiData(data.values, data.index, data.columns)
        else:
            return MultiData(data, index, columns)

    @staticmethod
    def concat(data_list: Union["SingleData"], axis=0) -> "MultiData":
        """concat all SingleData by index.
        TODO: now just for SingleData.

        Parameters
        ----------
        index_data_list : List[SingleData]
            the list of all SingleData to concat.

        Returns
        -------
        MultiData
            the MultiData with ndim == 2
        """
        if axis == 0:
            raise NotImplementedError(f"please implement this func when axis == 0")
        elif axis == 1:
            # get all index and row
            all_index = set()
            for index_data in data_list:
                all_index = all_index | set(index_data.index)
            all_index = list(all_index)
            all_index.sort()
            all_index_map = dict(zip(all_index, range(len(all_index))))

            # concat all
            tmp_data = np.full((len(all_index), len(data_list)), np.NaN)
            for data_id, index_data in enumerate(data_list):
                assert isinstance(index_data, SingleData)
                now_data_map = [all_index_map[index] for index in index_data.index]
                tmp_data[now_data_map, data_id] = index_data.data
            return MultiData(tmp_data, all_index)
        else:
            raise ValueError(f"axis must be 0 or 1")


class BaseData:
    """Base data structure of SingleData and MultiData."""

    def __init__(self):
        self.index_columns = self._get_index_columns()

    def _get_index_columns(self):
        index_columns = []
        if hasattr(self, "index"):
            index_columns.append(self.index)
        if hasattr(self, "columns"):
            index_columns.append(self.columns)
        return index_columns

    def _align_index(self, other):
        """Align index before performing the four arithmetic operations."""
        raise NotImplementedError(f"please implement _align_index func")

    def __add__(self, other):
        if isinstance(other, (int, float, np.number)):
            return self.__class__(self.data + other, *self.index_columns)
        elif isinstance(other, self.__class__):
            tmp_data1, tmp_data2 = self._align_index(other)
            return self.__class__(tmp_data1.data + tmp_data2.data, *tmp_data1.index_columns)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (int, float, np.number)):
            return self.__class__(self.data - other, *self.index_columns)
        elif isinstance(other, self.__class__):
            tmp_data1, tmp_data2 = self._align_index(other)
            return self.__class__(tmp_data1.data - tmp_data2.data, *tmp_data1.index_columns)
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float, np.number)):
            return self.__class__(other - self.data, *self.index_columns)
        elif isinstance(other, self.__class__):
            tmp_data1, tmp_data2 = self._align_index(other)
            return self.__class__(tmp_data2.data - tmp_data1.data, *tmp_data1.index_columns)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float, np.number)):
            return self.__class__(self.data * other, *self.index_columns)
        elif isinstance(other, self.__class__):
            tmp_data1, tmp_data2 = self._align_index(other)
            return self.__class__(tmp_data1.data * tmp_data2.data, *tmp_data1.index_columns)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float, np.number)):
            return self.__class__(self.data / other, *self.index_columns)
        elif isinstance(other, self.__class__):
            tmp_data1, tmp_data2 = self._align_index(other)
            return self.__class__(tmp_data1.data / tmp_data2.data, *tmp_data1.index_columns)
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, (int, float, np.number)):
            return self.__class__(self.data == other, *self.index_columns)
        elif isinstance(other, self.__class__):
            tmp_data1, tmp_data2 = self._align_index(other)
            return self.__class__(tmp_data1.data == tmp_data2.data, *tmp_data1.index_columns)
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, (int, float, np.number)):
            return self.__class__(self.data > other, *self.index_columns)
        elif isinstance(other, self.__class__):
            tmp_data1, tmp_data2 = self._align_index(other)
            return self.__class__(tmp_data1.data > tmp_data2.data, *tmp_data1.index_columns)
        else:
            return NotImplemented

    def __lt__(self, other):
        if isinstance(other, (int, float, np.number)):
            return self.__class__(self.data < other, *self.index_columns)
        elif isinstance(other, self.__class__):
            tmp_data1, tmp_data2 = self._align_index(other)
            return self.__class__(tmp_data1.data < tmp_data2.data, *tmp_data1.index_columns)
        else:
            return NotImplemented

    def __invert__(self):
        return self.__class__(~self.data, *self.index_columns)

    def abs(self):
        """get the abs of data except np.NaN."""
        tmp_data = np.absolute(self.data)
        return self.__class__(tmp_data, *self.index_columns)

    def astype(self, dtype):
        """change the type of data."""
        tmp_data = self.data.astype(dtype)
        return self.__class__(tmp_data, *self.index_columns)

    def replace(self, to_replace: dict):
        assert isinstance(to_replace, dict)
        tmp_data = self.data.copy()
        for num in to_replace:
            if num in tmp_data:
                tmp_data[tmp_data == num] = to_replace[num]
        return self.__class__(tmp_data, *self.index_columns)

    def apply(self, func: Callable):
        """apply a function to data."""
        tmp_data = func(self.data)
        return self.__class__(tmp_data, *self.index_columns)

    def __len__(self):
        """the length of the data.

        Returns
        -------
        int
            the length of the data.
        """
        return len(self.data)

    def sum(self, axis=None):
        if axis is None:
            return np.nansum(self.data)
        elif axis == 0:
            tmp_data = np.nansum(self.data, axis=0)
            return SingleData(tmp_data, self.columns)
        elif axis == 1:
            tmp_data = np.nansum(self.data, axis=1)
            return SingleData(tmp_data, self.index)
        else:
            raise ValueError(f"axis must be None, 0 or 1")

    def mean(self, axis=None):
        if axis is None:
            return np.nanmean(self.data)
        elif axis == 0:
            tmp_data = np.nanmean(self.data, axis=0)
            return SingleData(tmp_data, self.columns)
        elif axis == 1:
            tmp_data = np.nanmean(self.data, axis=1)
            return SingleData(tmp_data, self.index)
        else:
            raise ValueError(f"axis must be None, 0 or 1")

    def count(self):
        return len(self.data[~np.isnan(self.data)])

    @property
    def empty(self):
        return len(self.data) == 0

    @property
    def values(self):
        return self.data


class SingleData(BaseData):
    def __init__(self, data: Union[int, float, np.number, list] = [], index: Union[list, pd.Index] = []):
        """A data structure of index and numpy data.
        It's used to replace pd.Series due to high-speed.

        Parameters
        ----------
        data : Union[int, float, np.floating, list, np.ndarray]
            the dim of data must be 1.
        index : Union[list, pd.Index]
            the index of data.
        """
        # data
        if isinstance(data, (int, float, np.floating)):
            self.data = np.full(len(index), fill_value=data, dtype=np.float64)
        elif isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            raise ValueError(f"data must be list or np.ndarray")
        # data in SingleData must be one dim
        assert self.data.ndim == 1
        # replace int with float
        if self.data.dtype == np.signedinteger:
            self.data = self.data.astype(np.float64)
        # replace None with np.NaN, because pd.Series does it.
        if None in self.data:
            self.data[self.data == None] = np.NaN

        # index
        if isinstance(index, list):
            if index == [] and len(self.data) > 0:
                index = list(range(len(self.data)))
            self.index = index
        elif isinstance(index, pd.Index):
            self.index = list(index)
        else:
            raise ValueError(f"index must be list or pd.Index")
        assert len(self.data) == len(self.index)
        # if data is not empty,
        self.index_map = dict(zip(self.index, range(len(self.index))))

        super(SingleData, self).__init__()

    def _align_index(self, other):
        if self.index == other.index:
            return self, other
        elif set(self.index) == set(other.index):
            return self, other.reindex(self.index)
        else:
            raise ValueError(
                f"The indexes of self and other do not meet the requirements of the four arithmetic operations"
            )

    def reindex(self, index, fill_value=np.NaN):
        """reindex data and fill the missing value with np.NaN.

        Parameters
        ----------
        new_index : list
            new index

        Returns
        -------
        SingleData
            reindex data
        """
        if self.index == index:
            return self
        tmp_data = np.full(len(index), fill_value, dtype=np.float64)
        for index_id, index_item in enumerate(index):
            if index_item in self.index:
                tmp_data[index_id] = self.data[self.index_map[index_item]]
        return SingleData(tmp_data, index)

    def add(self, other, fill_value=0):
        common_index = list(set(self.index) | set(other.index))
        tmp_data1 = self.reindex(common_index, fill_value)
        tmp_data2 = other.reindex(common_index, fill_value)
        return tmp_data1 + tmp_data2

    def to_dict(self):
        """convert SingleData to dict.

        Returns
        -------
        dict
            data with the dict format.
        """
        return dict(zip(self.index, self.data.tolist()))

    def to_series(self):
        return pd.Series(self.data, index=self.index)

    def __getitem__(self, index: Union["SingleData", int, str]):
        if isinstance(index, int):
            return self.data[index]
        elif isinstance(index, str):
            return self.data[self.index_map[index]]
        elif isinstance(index, SingleData):
            new_data = self.data[index.data]
            new_index = list(np.array(self.index)[index.data])
            return SingleData(new_data, new_index)
        else:
            raise ValueError(f"index must be SingleData, int, str")


class MultiData(BaseData):
    def __init__(
        self,
        data: Union[list, np.ndarray] = [[]],
        index: Union[list, pd.Index] = [],
        columns: Union[list, pd.Index] = [],
    ):
        """A data structure of index and numpy data.
        It's used to replace pd.DataFrame due to high-speed.

        Parameters
        ----------
        data : Union[list, np.ndarray]
            the dim of data must be 2.
        index : Union[list, pd.Index]
            the index of data.
        columns: Union[list, pd.Index]
            the columns of data.
        """
        # data
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            raise ValueError(f"data must be list or np.ndarray")
        # data in SingleData must be two dim
        assert self.data.ndim == 2
        # replace int with float
        if self.data.dtype == np.signedinteger:
            self.data = self.data.astype(np.float64)
        # replace None with np.NaN, because pd.DataFrame does it.
        if None in self.data:
            self.data[self.data == None] = np.NaN

        # index
        if isinstance(index, list):
            if index == [] and self.data.shape[0] > 0:
                index = list(range(self.data.shape[0]))
            self.index = index
        elif isinstance(index, pd.Index):
            self.index = list(index)
        else:
            raise ValueError(f"index must be list or pd.Index")
        assert self.data.shape[0] == len(self.index)
        # if data is not empty,
        self.index_map = dict(zip(self.index, range(len(self.index))))

        # columns
        if isinstance(columns, list):
            if columns == [] and self.data.shape[1] > 0:
                columns = list(range(self.data.shape[1]))
            self.columns = columns
        elif isinstance(columns, pd.Index):
            self.columns = list(columns)
        else:
            raise ValueError(f"columns must be list or pd.Index")
        assert self.data.shape[1] == len(self.columns)
        # if data is not empty,
        self.columns_map = dict(zip(self.columns, range(len(self.columns))))

        super(MultiData, self).__init__()

    def _align_index(self, other):
        if self.index_columns == other.index_columns:
            return self, other
        else:
            raise ValueError(
                f"The indexes of self and other do not meet the requirements of the four arithmetic operations"
            )

    def __getitem__(self, col) -> SingleData:
        if col not in self.columns:
            return SingleData()
        else:
            return SingleData(self.data[:, self.columns_map[col]], self.index)

    def loc(self, start, end, col=None):
        start_id = bisect.bisect_left(self.index, start)
        end_id = bisect.bisect_right(self.index, end)
        if col is None:
            return MultiData(self.data[start_id:end_id], self.index[start_id:end_id], self.columns)
        else:
            return SingleData(self.data[start_id:end_id, self.columns_map[col]], self.index[start_id:end_id])

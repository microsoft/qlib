# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Motivation of index_data
- Pandas has a lot of user-friendly interfaces. However, integrating too much features in a single tool bring to much overhead and makes it much slower than numpy.
    Some users just want a simple numpy dataframe with indices and don't want such a complicated tools.
    Such users are the target of `index_data`

`index_data` try to behave like pandas (some API will be different because we try to be simpler and more intuitive) but don't compromize the performance. It provides the basic numpy data and simple indexing feature. If users call APIs which may compromize the performance, index_data will raise Errors.
"""

from typing import Tuple, Union, Callable, List
import bisect

import numpy as np
import pandas as pd


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


class Index:
    """
    This is for indexing(rows or columns)

    Read-only operations has higher priorities than others.
    So this class is designed in a **read-only** way to shared data for queries.
    Modifications will results in new Index.

    NOTE: the indexing has following flaws
    - duplicated index value is not well supported (only the first appearance will be considered)
    - The order of the index is not considered!!!! So the slicing will not behave like pandas when indexings are ordered
    """
    def __init__(self, idx_list: Union[List, pd.Index, "Index", int]):
        self.idx_list: np.ndarray = None  # using array type for index list will make things easier
        if isinstance(idx_list, Index):
            # Fast read-only copy
            self.idx_list = idx_list.idx_list
            self.index_map = idx_list.index_map
            self._is_sorted = idx_list._is_sorted
        elif isinstance(idx_list, int):
            self.index_map = self.idx_list = np.arange(idx_list)
            self._is_sorted = True
        else:
            self.idx_list = np.array(idx_list)
            # NOTE: only the first appearance is indexed
            self.index_map = dict(zip(self.idx_list, range(len(self))))
            self._is_sorted = False

    def __getitem__(self, i: int):
        return self.idx_list[i]

    def index(self, item) -> int:
        """
        Given the index value, get the integer index

        """
        return self.index_map[item]

    def __eq__(self, other: "Index"):
        # NOTE:  np.nan is not supported in the index
        return (self.idx_list == other.idx_list).all()

    def __len__(self):
        return len(self.idx_list)

    def is_sorted(self):
        return self._is_sorted

    def sort(self) -> Tuple["Index", np.ndarray]:
        """
        sort the index

        Returns
        -------
        Tuple["Index", np.ndarray]:
            the sorted Index and the changed index
        """
        sorted_idx = np.argsort(self.idx_list)
        idx = Index(self.idx_list[sorted_idx])
        idx._is_sorted = True
        return idx, sorted_idx



class LocIndexer:
    """
    `Indexer` will behave like the `LocIndexer` in Pandas

    Read-only operations has higher priorities than others.
    So this class is designed in a read-only way to shared data for queries.
    Modifications will results in new Index.
    """
    def __init__(self, index_data: "IndexData", indices: List[Index], int_loc: bool = False):
        self._indices: List[Index] = indices
        self._bind_id = index_data  # bind index data
        self._int_loc = int_loc
        assert self._bind_id.data.ndim == len(self._indices)

    @staticmethod
    def proc_idx_l(indices: List[Union[List, pd.Index, Index]], data_shape: Tuple = None) -> List[Index]:
        """ process the indices from user and output a list of `Index` """
        res = []
        for i, idx in enumerate(indices):
            res.append(Index(data_shape[i] if len(idx) == 0 else idx))
        return res

    def _slc_convert(self, index: Index, indexing: slice) -> slice:
        """
        convert value-based indexing to integer-based indexing.

        Parameters
        ----------
        index : Index
            index data.
        indexing : slice
            value based indexing data with slice type for indexing.

        Returns
        -------
        slice:
            the integer based slicing
        """
        if index.is_sorted():
            int_start = None if indexing.start is None else bisect.bisect_left(index, indexing.start)
            int_stop = None if indexing.stop is None else bisect.bisect_right(index, indexing.stop)
        else:
            int_start = None if indexing.start is None else index.index(indexing.start)
            int_stop = None if indexing.stop is None else index.index(indexing.stop) + 1
        return slice(int_start, int_stop)

    def __getitem__(self, indexing):
        """

        Parameters
        ----------
        indexing :
            query for data

        Raises
        ------
        KeyError:
            If the non-slice index is queried but does not exist, `KeyError` is raised.
        """
        # 1) convert slices to int loc
        if not isinstance(indexing, tuple):
            # NOTE: tuple is not supported for indexing
            indexing = (indexing, )

        # TODO: create a subclass for single value query
        assert len(indexing) <= len(self._indices)

        int_indexing = []
        for dim, index in enumerate(self._indices):
            if dim < len(indexing):
                _indexing = indexing[dim]
                if not self._int_loc:  # type converting is only necessary when it is not `iloc`
                    if isinstance(_indexing, slice):
                        _indexing = self._slc_convert(index, _indexing)
                    elif isinstance(_indexing, (IndexData, np.ndarray)):
                        if isinstance(_indexing, IndexData):
                            _indexing = _indexing.data
                        assert _indexing.ndim == 1
                        if _indexing.dtype != np.bool:
                            _indexing = np.array(list(index.index(i) for i in _indexing))
                    else:
                        _indexing = index.index(_indexing)
            else:
                _indexing = slice(None)
            int_indexing.append(_indexing)

        # 2) select data and index
        new_data = self._bind_id.data[tuple(int_indexing)]
        new_indices = [idx[indexing] for idx, indexing in zip(self._indices, int_indexing)]

        # 3) squash dimensions
        new_indices = [idx for idx in new_indices if isinstance(idx, np.ndarray) and idx.ndim > 0] # squash the zero dim indexing

        if new_data.ndim == 0:
            return new_data
        else:
            if new_data.ndim == 1:
                cls = SingleData
            elif new_data.ndim == 2:
                cls = MultiData
            else:
                raise ValueError("Not supported")
            return cls(new_data, *new_indices)


class IndexData:
    """
    Base data structure of SingleData and MultiData.

    NOTE:
    - For performance issue, only **np.floating** is supported in the underlayer data !!!
    - Boolean based on np.floating is also supported. Here are some examples

    .. code-block:: python

        np.array([ np.nan]).any() -> True
        np.array([ np.nan]).all() -> True
        np.array([1. , 0.]).any() -> True
        np.array([1. , 0.]).all() -> False
    """

    loc_idx_cls = LocIndexer
    def __init__(self, data: np.ndarray, *indices: Union[List, pd.Index, Index]):

        self.data = data
        self.indices = indices

        # get the expected data shape
        # - The index has higher priority
        self.data = np.array(data)

        expected_dim = max(self.data.ndim, len(indices))

        data_shape = []
        for i in range(expected_dim):
            idx_l = indices[i] if len(indices) > i else []
            if len(idx_l) == 0:
                data_shape.append(self.data.shape[i])
            else:
                data_shape.append(len(idx_l))
        data_shape = tuple(data_shape)

        # broadcast the data to expected shape
        self.data = np.broadcast_to(self.data, data_shape)

        self.data = self.data.astype(np.float64)
        # Please notice following cases when converting the type
        # - np.array([None, 1]).astype(np.float64) -> array([nan,  1.])

        # create index from user's index data.
        self.indices: List[Index] = self.loc_idx_cls.proc_idx_l(indices, data_shape)

        for dim in range(expected_dim):
            assert self.data.shape[dim] == len(self.indices[dim])

        self.ndim = expected_dim

    # indexing related methods
    @property
    def loc(self):
        return self.loc_idx_cls(index_data=self, indices=self.indices)

    @property
    def iloc(self):
        return self.loc_idx_cls(index_data=self, indices=self.indices, int_loc=True)

    @property
    def index(self):
        return self.indices[0]

    @property
    def columns(self):
        return self.indices[1]

    def _align_indices(self, other):
        """Align index before performing the four arithmetic operations."""
        raise NotImplementedError(f"please implement _align_indices func")

    def sort_index(self, axis=0, inplace=True):
        assert inplace, "Only support sorting inplace now"
        self.indices[axis], sorted_idx = self.indices[axis].sort()
        self.data = np.take(self.data, sorted_idx, axis=axis)

    # calculation related methods
    def __getattribute__(self, attr_name: str):
        # 1) use a unified operation for the basic operation

        def _basic_binary_ops(other):
            self_data_method = getattr(self.data, attr_name)

            if isinstance(other, (int, float, np.number)):
                return self.__class__(self_data_method(other))
            elif isinstance(other, self.__class__):
                # TODO: bad interface
                tmp_data1, tmp_data2 = self._align_indices(other)
                return self.__class__(self_data_method(tmp_data2.data), *self.indices)
            else:
                return NotImplemented

        if attr_name in {"__add__", "__sub__", "__rsub__", "__mul__", "__truediv__", "__eq__", "__gt__", "__lt__"}:
            return _basic_binary_ops

        # 2) otherwise, follow the default behavior
        return super().__getattribute__(attr_name)

    # The code below could be simpler like methods in __getattribute__
    def __invert__(self):
        return self.__class__(~self.data.astype(np.bool), *self.indices)

    def abs(self):
        """get the abs of data except np.NaN."""
        tmp_data = np.absolute(self.data)
        return self.__class__(tmp_data, *self.indices)

    def replace(self, to_replace: dict):
        assert isinstance(to_replace, dict)
        tmp_data = self.data.copy()
        for num in to_replace:
            if num in tmp_data:
                tmp_data[tmp_data == num] = to_replace[num]
        return self.__class__(tmp_data, *self.indices)

    def apply(self, func: Callable):
        """apply a function to data."""
        tmp_data = func(self.data)
        return self.__class__(tmp_data, *self.indices)

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

    def isna(self):
        return self.__class__(np.isnan(self.data), *self.indices)

    def count(self):
        return len(self.data[~np.isnan(self.data)])

    @property
    def empty(self):
        return len(self.data) == 0

    @property
    def values(self):
        return self.data


class SingleData(IndexData):
    def __init__(self, data: Union[int, float, np.number, list, dict, pd.Series] = [], index: Union[List, pd.Index, Index] = []):
        """A data structure of index and numpy data.
        It's used to replace pd.Series due to high-speed.

        Parameters
        ----------
        data : Union[int, float, np.number, list, dict, pd.Series]
            the input data
        index : Union[list, pd.Index]
            the index of data.
            empty list indicates that auto filling the index to the length of data
        """
        # for special data type
        if isinstance(data, dict):
            assert len(index) == 0
            index, data = zip(*data.items())
        elif isinstance(data, pd.Series):
            assert len(index) == 0
            index, data = data.index, data.values
        super().__init__(data, index)
        assert self.ndim == 1

    def _align_indices(self, other):
        if self.index == other.index:
            return self, other
        elif set(self.index) == set(other.index):
            return self, other.reindex(self.index)
        else:
            raise ValueError(
                f"The indexes of self and other do not meet the requirements of the four arithmetic operations")

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
        # TODO: This method can be more general
        if self.index == index:
            return self
        tmp_data = np.full(len(index), fill_value, dtype=np.float64)
        for index_id, index_item in enumerate(index):
            if index_item in self.index:
                tmp_data[index_id] = self.data[self.index_map[index_item]]
        return SingleData(tmp_data, index)

    def add(self, other, fill_value=0):
        # TODO: add and __add__ are a little confusing.
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

    def __repr__(self) -> str:
        return str(pd.Series(self.data, index=self.index))


class MultiData(IndexData):
    def __init__(self,
                 data: Union[int, float, np.number, list] = [],
                 index: Union[List, pd.Index, Index] = [],
                 columns: Union[List, pd.Index, Index] = []):
        """A data structure of index and numpy data.
        It's used to replace pd.DataFrame due to high-speed.

        Parameters
        ----------
        data : Union[list, np.ndarray]
            the dim of data must be 2.
        index : Union[List, pd.Index, Index]
            the index of data.
        columns: Union[List, pd.Index, Index]
            the columns of data.
        """
        if isinstance(data, pd.DataFrame):
            index, columns, data = data.index, data.columns, data.values
        super().__init__(data, index, columns)
        assert self.ndim == 2

    def _align_indices(self, other):
        if self.index_columns == other.index_columns:
            return self, other
        else:
            raise ValueError(
                f"The indexes of self and other do not meet the requirements of the four arithmetic operations")

    def __repr__(self) -> str:
        return str(pd.DataFrame(self.data, index=self.index, columns=self.columns))

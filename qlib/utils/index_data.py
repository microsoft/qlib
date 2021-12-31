# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Motivation of index_data
- Pandas has a lot of user-friendly interfaces. However, integrating too much features in a single tool bring to much overhead and makes it much slower than numpy.
    Some users just want a simple numpy dataframe with indices and don't want such a complicated tools.
    Such users are the target of `index_data`

`index_data` try to behave like pandas (some API will be different because we try to be simpler and more intuitive) but don't compromise the performance. It provides the basic numpy data and simple indexing feature. If users call APIs which may compromise the performance, index_data will raise Errors.
"""

from typing import Dict, Tuple, Union, Callable, List
import bisect

import numpy as np
import pandas as pd


def concat(data_list: Union["SingleData"], axis=0) -> "MultiData":
    """concat all SingleData by index.
    TODO: now just for SingleData.

    Parameters
    ----------
    data_list : List[SingleData]
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


def sum_by_index(data_list: Union["SingleData"], new_index: list, fill_value=0) -> "SingleData":
    """concat all SingleData by new index.

    Parameters
    ----------
    data_list : List[SingleData]
        the list of all SingleData to sum.
    new_index : list
        the new_index of new SingleData.
    fill_value : float
        fill the missing values ​​or replace np.NaN.

    Returns
    -------
    SingleData
        the SingleData with new_index and values after sum.
    """
    data_list = [data.to_dict() for data in data_list]
    data_sum = {}
    for id in new_index:
        item_sum = 0
        for data in data_list:
            if id in data and not np.isnan(data[id]):
                item_sum += data[id]
            else:
                item_sum += fill_value
        data_sum[id] = item_sum
    return SingleData(data_sum)


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

    def _convert_type(self, item):
        """

        After user creates indices with Type A, user may query data with other types with the same info.
            This method try to make type conversion and make query sane rather than raising KeyError strictly

        Parameters
        ----------
        item :
            The item to query index
        """

        if self.idx_list.dtype.type is np.datetime64:
            if isinstance(item, pd.Timestamp):
                # This happens often when creating index based on pandas.DatetimeIndex and query with pd.Timestamp
                return item.to_numpy()
        return item

    def index(self, item) -> int:
        """
        Given the index value, get the integer index

        Parameters
        ----------
        item :
            The item to query

        Returns
        -------
        int:
            The index of the item

        Raises
        ------
        KeyError:
            If the query item does not exist
        """
        try:
            return self.index_map[self._convert_type(item)]
        except IndexError:
            raise KeyError(f"{item} can't be found in {self}")

    def __or__(self, other: "Index"):
        return Index(idx_list=list(set(self.idx_list) | set(other.idx_list)))

    def __eq__(self, other: "Index"):
        # NOTE:  np.nan is not supported in the index
        if self.idx_list.shape != other.idx_list.shape:
            return False
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

    def tolist(self):
        """return the index with the format of list."""
        return self.idx_list.tolist()


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
        """process the indices from user and output a list of `Index`"""
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
            indexing = (indexing,)

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
                # Default to select all when user input is not given
                _indexing = slice(None)
            int_indexing.append(_indexing)

        # 2) select data and index
        new_data = self._bind_id.data[tuple(int_indexing)]
        # return directly if it is scalar
        if new_data.ndim == 0:
            return new_data
        # otherwise we go on to the index part
        new_indices = [idx[indexing] for idx, indexing in zip(self._indices, int_indexing)]

        # 3) squash dimensions
        new_indices = [
            idx for idx in new_indices if isinstance(idx, np.ndarray) and idx.ndim > 0
        ]  # squash the zero dim indexing

        if new_data.ndim == 1:
            cls = SingleData
        elif new_data.ndim == 2:
            cls = MultiData
        else:
            raise ValueError("Not supported")
        return cls(new_data, *new_indices)


class BinaryOps:
    def __init__(self, method_name):
        self.method_name = method_name

    def __get__(self, obj, *args):
        # bind object
        self.obj = obj
        return self

    def __call__(self, other):
        self_data_method = getattr(self.obj.data, self.method_name)

        if isinstance(other, (int, float, np.number)):
            return self.obj.__class__(self_data_method(other), *self.obj.indices)
        elif isinstance(other, self.obj.__class__):
            other_aligned = self.obj._align_indices(other)
            return self.obj.__class__(self_data_method(other_aligned.data), *self.obj.indices)
        else:
            return NotImplemented


def index_data_ops_creator(*args, **kwargs):
    """
    meta class for auto generating operations for index data.
    """
    for method_name in ["__add__", "__sub__", "__rsub__", "__mul__", "__truediv__", "__eq__", "__gt__", "__lt__"]:
        args[2][method_name] = BinaryOps(method_name=method_name)
    return type(*args)


class IndexData(metaclass=index_data_ops_creator):
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
        if self.data.shape != data_shape:
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

    def __getitem__(self, args):
        # NOTE: this tries to behave like a numpy array to be compatible with numpy aggregating function like nansum and nanmean
        return self.iloc[args]

    def _align_indices(self, other: "IndexData") -> "IndexData":
        """
        Align all indices of `other` to `self` before performing the arithmetic operations.
        This function will return a new IndexData rather than changing data in `other` inplace

        Parameters
        ----------
        other : "IndexData"
            the index in `other` is to be changed

        Returns
        -------
        IndexData:
            the data in `other` with index aligned to `self`
        """
        raise NotImplementedError(f"please implement _align_indices func")

    def sort_index(self, axis=0, inplace=True):
        assert inplace, "Only support sorting inplace now"
        self.indices[axis], sorted_idx = self.indices[axis].sort()
        self.data = np.take(self.data, sorted_idx, axis=axis)

    # The code below could be simpler like methods in __getattribute__
    def __invert__(self):
        return self.__class__(~self.data.astype(np.bool), *self.indices)

    def abs(self):
        """get the abs of data except np.NaN."""
        tmp_data = np.absolute(self.data)
        return self.__class__(tmp_data, *self.indices)

    def replace(self, to_replace: Dict[np.number, np.number]):
        assert isinstance(to_replace, dict)
        tmp_data = self.data.copy()
        for num in to_replace:
            if num in tmp_data:
                tmp_data[self.data == num] = to_replace[num]
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

    def sum(self, axis=None, dtype=None, out=None):
        assert out is None and dtype is None, "`out` is just for compatible with numpy's aggregating function"
        # FIXME: weird logic and not general
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

    def mean(self, axis=None, dtype=None, out=None):
        assert out is None and dtype is None, "`out` is just for compatible with numpy's aggregating function"
        # FIXME: weird logic and not general
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

    def fillna(self, value=0.0, inplace: bool = False):
        if inplace:
            self.data = np.nan_to_num(self.data, nan=value)
        else:
            return self.__class__(np.nan_to_num(self.data, nan=value), *self.indices)

    def count(self):
        return len(self.data[~np.isnan(self.data)])

    def all(self):
        if None in self.data:
            return self.data[self.data is not None].all()
        else:
            return self.data.all()

    @property
    def empty(self):
        return len(self.data) == 0

    @property
    def values(self):
        return self.data


class SingleData(IndexData):
    def __init__(
        self, data: Union[int, float, np.number, list, dict, pd.Series] = [], index: Union[List, pd.Index, Index] = []
    ):
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
            if len(data) > 0:
                index, data = zip(*data.items())
            else:
                index, data = [], []
        elif isinstance(data, pd.Series):
            assert len(index) == 0
            index, data = data.index, data.values
        elif isinstance(data, (int, float, np.number)):
            data = [data]
        super().__init__(data, index)
        assert self.ndim == 1

    def _align_indices(self, other):
        if self.index == other.index:
            return other
        elif set(self.index) == set(other.index):
            return other.reindex(self.index)
        else:
            raise ValueError(
                f"The indexes of self and other do not meet the requirements of the four arithmetic operations"
            )

    def reindex(self, index: Index, fill_value=np.NaN):
        """reindex data and fill the missing value with np.NaN.

        Parameters
        ----------
        new_index : list
            new index
        fill_value:
            what value to fill if index is missing

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
            try:
                tmp_data[index_id] = self.loc[index_item]
            except KeyError:
                pass
        return SingleData(tmp_data, index)

    def add(self, other: "SingleData", fill_value=0):
        # TODO: add and __add__ are a little confusing.
        # This could be a more general
        common_index = self.index | other.index
        common_index, _ = common_index.sort()
        tmp_data1 = self.reindex(common_index, fill_value)
        tmp_data2 = other.reindex(common_index, fill_value)
        return tmp_data1.fillna(fill_value) + tmp_data2.fillna(fill_value)

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
    def __init__(
        self,
        data: Union[int, float, np.number, list] = [],
        index: Union[List, pd.Index, Index] = [],
        columns: Union[List, pd.Index, Index] = [],
    ):
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
        if self.indices == other.indices:
            return other
        else:
            raise ValueError(
                f"The indexes of self and other do not meet the requirements of the four arithmetic operations"
            )

    def __repr__(self) -> str:
        return str(pd.DataFrame(self.data, index=self.index, columns=self.columns))

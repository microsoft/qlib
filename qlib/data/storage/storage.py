# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
from typing import Iterable, overload, Tuple, List, Text, Union, Dict

import numpy as np
import pandas as pd
from qlib.log import get_module_logger

# calendar value type
CalVT = str

# instrument value
InstVT = List[Tuple[CalVT, CalVT]]
# instrument key
InstKT = Text

logger = get_module_logger("storage")

"""
If the user is only using it in `qlib`, you can customize Storage to implement only the following methods:

class UserCalendarStorage(CalendarStorage):

    @property
    def data(self):
        pass

class UserInstrumentStorage(InstrumentStorage):

    @property
    def data(self):
        pass

class UserFeatureStorage(FeatureStorage):

    @check_storage
    def __getitem__(self, i: slice) -> pd.Series:
        pass

"""


class StorageMeta(type):
    """unified management of raise when storage is not exists"""

    def __new__(cls, name, bases, dict):
        class_obj = type.__new__(cls, name, bases, dict)

        # The calls to __iter__ and __getitem__ do not pass through __getattribute__.
        # In order to throw an exception before calling __getitem__, use the metaclass
        _getitem_func = getattr(class_obj, "__getitem__")

        def _getitem(obj, item):
            _check_func = getattr(obj, "_check")
            if callable(_check_func):
                _check_func()
            return _getitem_func(obj, item)

        setattr(class_obj, "__getitem__", _getitem)
        return class_obj


class BaseStorage(metaclass=StorageMeta):
    @property
    def storage_name(self) -> str:
        return re.findall("[A-Z][^A-Z]*", self.__class__.__name__)[-2]

    def check_exists(self) -> bool:
        """check if storage(uri) exists, if not exists: return False"""
        raise NotImplementedError("Subclass of BaseStorage must implement `check_exists` method")

    def clear(self) -> None:
        """clear storage"""
        raise NotImplementedError("Subclass of BaseStorage must implement `clear` method")

    def __len__(self) -> 0:
        return len(self.data) if self.check_exists() else 0

    def __getitem__(self, item: Union[slice, Union[int, InstKT]]):
        raise NotImplementedError(
            "Subclass of BaseStorage must implement `__getitem__(i: Union[int, InstKT])`/`__getitem__(s: slice)`  method"
        )

    def _check(self):
        # check storage(uri)
        if not self.check_exists():
            parameters_info = [f"{_k}={_v}" for _k, _v in self.__dict__.items()]
            raise ValueError(f"{self.storage_name.lower()} not exists, storage parameters: {parameters_info}")

    def __getattribute__(self, item):
        if item == "data":
            self._check()
        return super(BaseStorage, self).__getattribute__(item)


class CalendarStorage(BaseStorage):
    """
    The behavior of CalendarStorage's methods and List's methods of the same name remain consistent
    """

    def __init__(self, freq: str, future: bool, uri: str, **kwargs):
        self.freq = freq
        self.future = future
        self.uri = uri

    @property
    def data(self) -> Iterable[CalVT]:
        """get all data"""
        raise NotImplementedError("Subclass of CalendarStorage must implement `data` method")

    def extend(self, iterable: Iterable[CalVT]) -> None:
        raise NotImplementedError("Subclass of CalendarStorage must implement `extend` method")

    def index(self, value: CalVT) -> int:
        raise NotImplementedError("Subclass of CalendarStorage must implement `index` method")

    def insert(self, index: int, value: CalVT) -> None:
        raise NotImplementedError("Subclass of CalendarStorage must implement `insert` method")

    def remove(self, value: CalVT) -> None:
        raise NotImplementedError("Subclass of CalendarStorage must implement `remove` method")

    @overload
    def __setitem__(self, i: int, value: CalVT) -> None:
        """x.__setitem__(i, o) <==> (x[i] = o)"""
        ...

    @overload
    def __setitem__(self, s: slice, value: Iterable[CalVT]) -> None:
        """x.__setitem__(s, o) <==> (x[s] = o)"""
        ...

    def __setitem__(self, i, value) -> None:
        raise NotImplementedError(
            "Subclass of CalendarStorage must implement `__setitem__(i: int, o: CalVT)`/`__setitem__(s: slice, o: Iterable[CalVT])`  method"
        )

    @overload
    def __delitem__(self, i: int) -> None:
        """x.__delitem__(i) <==> del x[i]"""
        ...

    @overload
    def __delitem__(self, i: slice) -> None:
        """x.__delitem__(slice(start: int, stop: int, step: int)) <==> del x[start:stop:step]"""
        ...

    def __delitem__(self, i) -> None:
        raise NotImplementedError(
            "Subclass of CalendarStorage must implement `__delitem__(i: int)`/`__delitem__(s: slice)`  method"
        )

    @overload
    def __getitem__(self, s: slice) -> Iterable[CalVT]:
        """x.__getitem__(slice(start: int, stop: int, step: int)) <==> x[start:stop:step]"""
        ...

    @overload
    def __getitem__(self, i: int) -> CalVT:
        """x.__getitem__(i) <==> x[i]"""
        ...

    def __getitem__(self, i) -> CalVT:
        raise NotImplementedError(
            "Subclass of CalendarStorage must implement `__getitem__(i: int)`/`__getitem__(s: slice)`  method"
        )


class InstrumentStorage(BaseStorage):
    def __init__(self, market: str, uri: str, **kwargs):
        self.market = market
        self.uri = uri

    @property
    def data(self) -> Dict[InstKT, InstVT]:
        """get all data"""
        raise NotImplementedError("Subclass of InstrumentStorage must implement `data` method")

    def update(self, *args, **kwargs) -> None:
        """D.update([E, ]**F) -> None.  Update D from mapping/iterable E and F.
        If E present and has a .keys() method, does:     for k in E: D[k] = E[k]
        If E present and lacks .keys() method, does:     for (k, v) in E: D[k] = v
        In either case, this is followed by: for k, v in F.items(): D[k] = v
        """
        raise NotImplementedError("Subclass of InstrumentStorage must implement `update` method")

    def __setitem__(self, k: InstKT, v: InstVT) -> None:
        """Set self[key] to value."""
        raise NotImplementedError("Subclass of InstrumentStorage must implement `__setitem__` method")

    def __delitem__(self, k: InstKT) -> None:
        """Delete self[key]."""
        raise NotImplementedError("Subclass of InstrumentStorage must implement `__delitem__` method")

    def __getitem__(self, k: InstKT) -> InstVT:
        """x.__getitem__(k) <==> x[k]"""
        raise NotImplementedError("Subclass of InstrumentStorage must implement `__getitem__` method")


class FeatureStorage(BaseStorage):
    def __init__(self, instrument: str, field: str, freq: str, uri: str, **kwargs):
        self.instrument = instrument
        self.field = field
        self.freq = freq
        self.uri = uri

    @property
    def data(self) -> pd.Series:
        """get all data"""
        raise NotImplementedError("Subclass of FeatureStorage must implement `data` method")

    @property
    def start_index(self) -> Union[int, None]:
        """get FeatureStorage start index
        If len(self) == 0; return None
        """
        raise NotImplementedError("Subclass of FeatureStorage must implement `data` method")

    @property
    def end_index(self) -> Union[int, None]:
        if len(self) == 0:
            return None
        return None if len(self) == 0 else self.start_index + len(self) - 1

    def write(self, data_array: Union[List, np.ndarray, Tuple], index: int = None):
        """Write data_array to FeatureStorage starting from index.
        If index is None, append data_array to feature.
        If len(data_array) == 0; return
        If (index - self.end_index) >= 1, self[end_index+1: index] will be filled with np.nan


        Examples:

                feature:
                    3   4
                    4   5
                    5   6

            >>> self.write([6, 7], index=6)

                feature:
                    3   4
                    4   5
                    5   6
                    6   6
                    7   7

            >>> self.write([8], index=9)

                feature:
                    3   4
                    4   5
                    5   6
                    6   6
                    7   7
                    8   np.nan
                    9   8

            >>> self.write([1, np.nan], index=3)

                feature:
                    3   1
                    4   np.nan
                    5   6
                    6   6
                    7   7
                    8   np.nan
                    9   8

        """
        raise NotImplementedError("Subclass of FeatureStorage must implement `write` method")

    def rebase(self, start_index: int = None, end_index: int = None):
        """Rebase the start_index and end_index of the FeatureStorage.

        Examples:

                feature:
                    3   4
                    4   5
                    5   6

            >>> self.rebase(start_index=4)

                feature:
                    4   5
                    5   6

            >>> self.rebase(start_index=3)

                feature:
                    3   np.nan
                    4   5
                    5   6

            >>> self.write([3], index=3)

                feature:
                    3   3
                    4   5
                    5   6

            >>> self.rebase(end_index=4)

                feature:
                    3   3
                    4   5

            >>> self.write([6, 7, 8], index=4)

                feature:
                    3   3
                    4   6
                    5   7
                    6   8

            >>> self.rebase(start_index=4, end_index=5)

                feature:
                    4   6
                    5   7

        """
        if start_index is None and end_index is None:
            logger.warning("both start_index and end_index are None, rebase is ignored")
            return

        if start_index < 0 or end_index < 0:
            logger.warning("start_index or end_index cannot be less than 0")
            return
        if start_index > end_index:
            logger.warning(
                f"start_index({start_index}) > end_index({end_index}), rebase is ignored; "
                f"if you need to clear the FeatureStorage, please execute: FeatureStorage.clear"
            )
            return

        start_index = self.start_index if start_index is None else end_index
        end_index = self.end_index if end_index is None else end_index
        if start_index <= self.start_index:
            self.write([np.nan] * (self.start_index - start_index), start_index)
        else:
            self.rewrite(self[start_index:].values, start_index)

        if end_index >= self.end_index:
            self.write([np.nan] * (end_index - self.end_index))
        else:
            self.rewrite(self[: end_index + 1].values, self.start_index)

    def rewrite(self, data: Union[List, np.ndarray, Tuple], index: int):
        """overwrite all data in FeatureStorage with data

        Parameters
        ----------
        data: Union[List, np.ndarray, Tuple]
            data
        index: int
            data start index
        """
        self.clear()
        self.write(data, index)

    @overload
    def __getitem__(self, s: slice) -> pd.Series:
        """x.__getitem__(slice(start: int, stop: int, step: int)) <==> x[start:stop:step]

        Returns
        -------
            pd.Series(values, index=pd.RangeIndex(start, len(values))
        """
        ...

    @overload
    def __getitem__(self, i: int) -> Tuple[int, float]:
        """x.__getitem__(y) <==> x[y]"""
        ...

    def __getitem__(self, i) -> Union[Tuple[int, float], pd.Series]:
        """x.__getitem__(y) <==> x[y]"""
        raise NotImplementedError(
            "Subclass of FeatureStorage must implement `__getitem__(i: int)`/`__getitem__(s: slice)` method"
        )

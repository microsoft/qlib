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
    def data(self) -> Iterable[CalVT]:
        '''get all data

        Raises
        ------
        ValueError
            If the data(storage) does not exist, raise ValueError
        '''
        raise NotImplementedError("Subclass of CalendarStorage must implement `data` method")


class UserInstrumentStorage(InstrumentStorage):

    @property
    def data(self) -> Dict[InstKT, InstVT]:
        '''get all data

        Raises
        ------
        ValueError
            If the data(storage) does not exist, raise ValueError
        '''
        raise NotImplementedError("Subclass of InstrumentStorage must implement `data` method")


class UserFeatureStorage(FeatureStorage):

    def __getitem__(self, s: slice) -> pd.Series:
        '''x.__getitem__(slice(start: int, stop: int, step: int)) <==> x[start:stop:step]

        Returns
        -------
            pd.Series(values, index=pd.RangeIndex(start, len(values))

        Notes
        -------
        if data(storage) does not exist:
            if isinstance(i, int):
                return (None, None)
            if isinstance(i,  slice):
                # return empty pd.Series
                return pd.Series(dtype=np.float32)
        '''
        raise NotImplementedError(
            "Subclass of FeatureStorage must implement `__getitem__(s: slice)` method"
        )


"""


class BaseStorage:
    @property
    def storage_name(self) -> str:
        return re.findall("[A-Z][^A-Z]*", self.__class__.__name__)[-2].lower()


class CalendarStorage(BaseStorage):
    """
    The behavior of CalendarStorage's methods and List's methods of the same name remain consistent
    """

    def __init__(self, freq: str, future: bool, **kwargs):
        self.freq = freq
        self.future = future
        self.kwargs = kwargs

    @property
    def data(self) -> Iterable[CalVT]:
        """get all data

        Raises
        ------
        ValueError
            If the data(storage) does not exist, raise ValueError
        """
        raise NotImplementedError("Subclass of CalendarStorage must implement `data` method")

    def clear(self) -> None:
        raise NotImplementedError("Subclass of CalendarStorage must implement `clear` method")

    def extend(self, iterable: Iterable[CalVT]) -> None:
        raise NotImplementedError("Subclass of CalendarStorage must implement `extend` method")

    def index(self, value: CalVT) -> int:
        """
        Raises
        ------
        ValueError
            If the data(storage) does not exist, raise ValueError
        """
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
        """
        Raises
        ------
        ValueError
            If the data(storage) does not exist, raise ValueError
        """
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
        """

        Raises
        ------
        ValueError
            If the data(storage) does not exist, raise ValueError

        """
        raise NotImplementedError(
            "Subclass of CalendarStorage must implement `__getitem__(i: int)`/`__getitem__(s: slice)`  method"
        )

    def __len__(self) -> int:
        """

        Raises
        ------
        ValueError
            If the data(storage) does not exist, raise ValueError

        """
        raise NotImplementedError("Subclass of CalendarStorage must implement `__len__`  method")


class InstrumentStorage(BaseStorage):
    def __init__(self, market: str, freq: str, **kwargs):
        self.market = market
        self.freq = freq
        self.kwargs = kwargs

    @property
    def data(self) -> Dict[InstKT, InstVT]:
        """get all data

        Raises
        ------
        ValueError
            If the data(storage) does not exist, raise ValueError
        """
        raise NotImplementedError("Subclass of InstrumentStorage must implement `data` method")

    def clear(self) -> None:
        raise NotImplementedError("Subclass of InstrumentStorage must implement `clear` method")

    def update(self, *args, **kwargs) -> None:
        """D.update([E, ]**F) -> None.  Update D from mapping/iterable E and F.

        Notes
        ------
            If E present and has a .keys() method, does:     for k in E: D[k] = E[k]

            If E present and lacks .keys() method, does:     for (k, v) in E: D[k] = v

            In either case, this is followed by: for k, v in F.items(): D[k] = v

        """
        raise NotImplementedError("Subclass of InstrumentStorage must implement `update` method")

    def __setitem__(self, k: InstKT, v: InstVT) -> None:
        """Set self[key] to value."""
        raise NotImplementedError("Subclass of InstrumentStorage must implement `__setitem__` method")

    def __delitem__(self, k: InstKT) -> None:
        """Delete self[key].

        Raises
        ------
        ValueError
            If the data(storage) does not exist, raise ValueError
        """
        raise NotImplementedError("Subclass of InstrumentStorage must implement `__delitem__` method")

    def __getitem__(self, k: InstKT) -> InstVT:
        """x.__getitem__(k) <==> x[k]"""
        raise NotImplementedError("Subclass of InstrumentStorage must implement `__getitem__` method")

    def __len__(self) -> int:
        """

        Raises
        ------
        ValueError
            If the data(storage) does not exist, raise ValueError

        """
        raise NotImplementedError("Subclass of InstrumentStorage must implement `__len__`  method")


class FeatureStorage(BaseStorage):
    def __init__(self, instrument: str, field: str, freq: str, **kwargs):
        self.instrument = instrument
        self.field = field
        self.freq = freq
        self.kwargs = kwargs

    @property
    def data(self) -> pd.Series:
        """get all data

        Notes
        ------
        if data(storage) does not exist, return empty pd.Series: `return pd.Series(dtype=np.float32)`
        """
        raise NotImplementedError("Subclass of FeatureStorage must implement `data` method")

    @property
    def start_index(self) -> Union[int, None]:
        """get FeatureStorage start index

        Notes
        -----
        If the data(storage) does not exist, return None
        """
        raise NotImplementedError("Subclass of FeatureStorage must implement `start_index` method")

    @property
    def end_index(self) -> Union[int, None]:
        """get FeatureStorage end index

        Notes
        -----
        The  right index of the data range (both sides are closed)

            The next  data appending point will be  `end_index + 1`

        If the data(storage) does not exist, return None
        """
        raise NotImplementedError("Subclass of FeatureStorage must implement `end_index` method")

    def clear(self) -> None:
        raise NotImplementedError("Subclass of FeatureStorage must implement `clear` method")

    def write(self, data_array: Union[List, np.ndarray, Tuple], index: int = None):
        """Write data_array to FeatureStorage starting from index.

        Notes
        ------
            If index is None, append data_array to feature.

            If len(data_array) == 0; return

            If (index - self.end_index) >= 1, self[end_index+1: index] will be filled with np.nan

        Examples
        ---------
            .. code-block::

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

        start_index and end_index are closed intervals: [start_index, end_index]

        Examples
        ---------

            .. code-block::

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
        storage_si = self.start_index
        storage_ei = self.end_index
        if storage_si is None or storage_ei is None:
            raise ValueError("storage.start_index or storage.end_index is None, storage may not exist")

        start_index = storage_si if start_index is None else start_index
        end_index = storage_ei if end_index is None else end_index

        if start_index is None or end_index is None:
            logger.warning("both start_index and end_index are None, or storage does not exist; rebase is ignored")
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

        if start_index <= storage_si:
            self.write([np.nan] * (storage_si - start_index), start_index)
        else:
            self.rewrite(self[start_index:].values, start_index)

        if end_index >= self.end_index:
            self.write([np.nan] * (end_index - self.end_index))
        else:
            self.rewrite(self[: end_index + 1].values, start_index)

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
        """x.__getitem__(y) <==> x[y]

        Notes
        -------
        if data(storage) does not exist:
            if isinstance(i, int):
                return (None, None)
            if isinstance(i,  slice):
                # return empty pd.Series
                return pd.Series(dtype=np.float32)
        """
        raise NotImplementedError(
            "Subclass of FeatureStorage must implement `__getitem__(i: int)`/`__getitem__(s: slice)` method"
        )

    def __len__(self) -> int:
        """

        Raises
        ------
        ValueError
            If the data(storage) does not exist, raise ValueError

        """
        raise NotImplementedError("Subclass of FeatureStorage must implement `__len__`  method")

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Iterable, overload, Tuple, List, Text, Iterator, Union, Dict

import pandas as pd

# calendar value type
CalVT = str

# instrument value
InstVT = List[Tuple[CalVT, CalVT]]
# instrument key
InstKT = Text


class CalendarStorage:
    def __init__(self, freq: str, future: bool, uri: str):
        self.freq = freq
        self.future = future
        self.uri = uri

    @property
    def data(self) -> Iterable[CalVT]:
        """get all data"""
        raise NotImplementedError("Subclass of CalendarStorage must implement `data` method")

    def extend(self, iterable: Iterable[CalVT]) -> None:
        raise NotImplementedError("Subclass of CalendarStorage must implement `extend` method")

    def clear(self) -> None:
        raise NotImplementedError("Subclass of CalendarStorage must implement `clear` method")

    def index(self, value: CalVT) -> int:
        raise NotImplementedError("Subclass of CalendarStorage must implement `index` method")

    def insert(self, index: int, value: CalVT) -> None:
        raise NotImplementedError("Subclass of CalendarStorage must implement `insert` method")

    def remove(self, value: CalVT) -> None:
        raise NotImplementedError("Subclass of CalendarStorage must implement `remove` method")

    @overload
    def __setitem__(self, i: int, value: CalVT) -> None:
        """x.__setitem__(i, o) <==> x[i] = o"""
        ...

    @overload
    def __setitem__(self, s: slice, value: Iterable[CalVT]) -> None:
        """x.__setitem__(s, o) <==> x[s] = o"""
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

    def __len__(self) -> int:
        """x.__len__() <==> len(x)"""
        raise NotImplementedError("Subclass of CalendarStorage must implement `__len__` method")

    def __iter__(self):
        raise NotImplementedError("Subclass of CalendarStorage must implement `__iter__` method")


class InstrumentStorage:
    def __init__(self, market: str, uri: str):
        self.market = market
        self.uri = uri

    @property
    def data(self) -> Dict[InstKT, InstVT]:
        """get all data"""
        raise NotImplementedError("Subclass of InstrumentStorage must implement `data` method")

    def clear(self) -> None:
        raise NotImplementedError("Subclass of InstrumentStorage must implement `clear` method")

    def update(self, *args, **kwargs) -> None:
        """D.update([E, ]**F) -> None.  Update D from mapping/iterable E and F.
        If E present and has a .keys() method, does:     for k in E: D[k] = E[k]
        If E present and lacks .keys() method, does:     for (k, v) in E: D[k] = v
        In either case, this is followed by: for k, v in F.items(): D[k] = v
        """
        raise NotImplementedError("Subclass of InstrumentStorage must implement `update` method")

    def __setitem__(self, k: InstKT, v: InstVT) -> None:
        """ Set self[key] to value. """
        raise NotImplementedError("Subclass of InstrumentStorage must implement `__setitem__` method")

    def __delitem__(self, k: InstKT) -> None:
        """ Delete self[key]. """
        raise NotImplementedError("Subclass of InstrumentStorage must implement `__delitem__` method")

    def __getitem__(self, k: InstKT) -> InstVT:
        """ x.__getitem__(k) <==> x[k] """
        raise NotImplementedError("Subclass of InstrumentStorage must implement `__getitem__` method")

    def __len__(self) -> int:
        """ Return len(self). """
        raise NotImplementedError("Subclass of InstrumentStorage must implement `__len__` method")

    def __iter__(self) -> Iterator[InstKT]:
        """ Return iter(self). """
        raise NotImplementedError("Subclass of InstrumentStorage must implement `__iter__` method")


class FeatureStorage:
    def __init__(self, instrument: str, field: str, freq: str, uri: str):
        self.instrument = instrument
        self.field = field
        self.freq = freq
        self.uri = uri

    @property
    def data(self) -> pd.Series:
        """get all data"""
        raise NotImplementedError("Subclass of FeatureStorage must implement `data` method")

    def clear(self):
        """ Remove all items from FeatureStorage. """
        raise NotImplementedError("Subclass of FeatureStorage must implement `clear` method")

    def extend(self, series: pd.Series):
        """Extend feature by appending elements from the series.

        Examples:

                feature:
                    3   4
                    4   5
                    5   6

            >>> self.extend(pd.Series({7: 8, 9:10}))

                feature:
                    3   4
                    4   5
                    5   6
                    6   np.nan
                    7   8
                    9   10

        """
        raise NotImplementedError("Subclass of FeatureStorage must implement `extend` method")

    def rebase(self, series: pd.Series):
        """Rebase feature header from the series.

        Examples:

                feature:
                    3   4
                    4   5
                    5   6

            >>> self.rebase(pd.Series({1: 2}))

                feature:
                    1   2
                    2   np.nan
                    3   4
                    4   5
                    5   6

            >>> self.rebase(pd.Series({5: 6, 7: 8, 9: 10}))

                feature:
                    5   6
                    7   8
                    9   10

            >>> self.rebase(pd.Series({11: 12, 12: 13,}))

                feature:
                    11   12
                    12   13

        """
        raise NotImplementedError("Subclass of FeatureStorage must implement `rebase` method")

    @overload
    def __getitem__(self, s: slice) -> pd.Series:
        """x.__getitem__(slice(start: int, stop: int, step: int)) <==> x[start:stop:step] == pd.Series(values, index=pd.RangeIndex(start, len(values))"""
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

    def __len__(self) -> int:
        """len(feature) <==> feature.__len__() """
        raise NotImplementedError("Subclass of FeatureStorage must implement `__len__` method")

    def __iter__(self) -> Iterable[Tuple[int, float]]:
        """iter(feature)"""
        raise NotImplementedError("Subclass of FeatureStorage must implement `__iter__` method")

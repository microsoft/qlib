# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import abc

from typing import (
    Iterable,
    overload,
    TypeVar,
    Tuple,
    List,
    Text,
    Optional,
    AbstractSet,
    Mapping,
    Iterator,
)


# calendar value type
CalVT = TypeVar("CalVT")

# instrument value
InstVT = List[Tuple[CalVT, CalVT]]
# instrument key
InstKT = Text


FeatureVT = Tuple[int, float]


class CalendarStorage:
    def __init__(self, uri: str):
        self._uri = uri

    def append(self, obj: CalVT) -> None:
        """ Append object to the end of the CalendarStorage. """
        raise NotImplementedError("Subclass of CalendarStorage must implement `append` method")

    def clear(self):
        """ Remove all items from CalendarStorage. """
        raise NotImplementedError("Subclass of CalendarStorage must implement `clear` method")

    def extend(self, iterable: Iterable[CalVT]):
        """ Extend list by appending elements from the iterable. """
        raise NotImplementedError("Subclass of CalendarStorage must implement `extend` method")

    @overload
    @abc.abstractmethod
    def __getitem__(self, s: slice) -> Iterable[CalVT]:
        """x.__getitem__(slice(start: int, stop: int, step: int)) <==> x[start:stop:step]"""
        raise NotImplementedError("Subclass of CalendarStorage must implement `__getitem__(s: slice)` method")

    @abc.abstractmethod
    def __getitem__(self, i: int) -> CalVT:
        """x.__getitem__(y) <==> x[y]"""

        raise NotImplementedError("Subclass of CalendarStorage must implement `__getitem__(i: int)` method")

    @abc.abstractmethod
    def __iter__(self) -> Iterator[CalVT]:
        """ Implement iter(self). """
        raise NotImplementedError("Subclass of CalendarStorage must implement `__iter__` method")

    def __len__(self) -> int:
        raise NotImplementedError("Subclass of CalendarStorage must implement `__len__` method")


class InstrumentStorage:
    def __init__(self, uri: str):
        self._uri = uri

    def clear(self) -> None:
        """ D.clear() -> None.  Remove all items from D. """
        raise NotImplementedError("Subclass of InstrumentStorage must implement `clear` method")

    @abc.abstractmethod
    def get(self, k: InstKT) -> Optional[InstVT]:
        """D.get(k) -> InstV or None"""
        raise NotImplementedError("Subclass of InstrumentStorage must implement `get` method")

    @abc.abstractmethod
    def items(self) -> AbstractSet[Tuple[InstKT, InstVT]]:
        """ D.items() -> a set-like object providing a view on D's items """
        raise NotImplementedError("Subclass of InstrumentStorage must implement `items` method")

    @abc.abstractmethod
    def keys(self) -> AbstractSet[InstKT]:
        """ D.keys() -> a set-like object providing a view on D's keys """
        raise NotImplementedError("Subclass of InstrumentStorage must implement `keys` method")

    def update(self, e: Mapping[InstKT, InstVT] = None, **f: InstVT) -> None:
        """
        D.update([e, ]**f) -> None.  Update D from dict/iterable e and f.
        If e is present and has a .keys() method, then does:  for k in e: D[k] = e[k]
        If e is present and lacks a .keys() method, then does:  for k, v in e: D[k] = v
        In either case, this is followed by: for k in f:  D[k] = f[k]
        """
        raise NotImplementedError("Subclass of InstrumentStorage must implement `update` method")

    def __setitem__(self, k: InstKT, v: InstVT) -> None:
        """ Set self[key] to value. """
        raise NotImplementedError("Subclass of InstrumentStorage must implement `__setitem__` method")

    def __delitem__(self, k: InstKT) -> None:
        """ Delete self[key]. """
        raise NotImplementedError("Subclass of InstrumentStorage must implement `__delitem__` method")

    @abc.abstractmethod
    def __getitem__(self, k: InstKT) -> InstVT:
        """ x.__getitem__(y) <==> x[y] """
        raise NotImplementedError("Subclass of InstrumentStorage must implement `__getitem__` method")

    def __len__(self) -> int:
        """ Return len(self). """
        raise NotImplementedError("Subclass of InstrumentStorage must implement `__len__` method")


class FeatureStorage:
    def __init__(self, uri: str):
        self._uri = uri

    def append(self, obj: FeatureVT) -> None:
        """ Append object to the end of the FeatureStorage. """
        raise NotImplementedError("Subclass of FeatureStorage must implement `append` method")

    def clear(self):
        """ Remove all items from FeatureStorage. """
        raise NotImplementedError("Subclass of FeatureStorage must implement `clear` method")

    def extend(self, iterable: Iterable[FeatureVT]):
        """ Extend list by appending elements from the iterable. """
        raise NotImplementedError("Subclass of FeatureStorage must implement `extend` method")

    @overload
    @abc.abstractmethod
    def __getitem__(self, s: slice) -> Iterable[FeatureVT]:
        """x.__getitem__(slice(start: int, stop: int, step: int)) <==> x[start:stop:step]"""
        raise NotImplementedError("Subclass of FeatureStorage must implement `__getitem__(s: slice)` method")

    @abc.abstractmethod
    def __getitem__(self, i: int) -> float:
        """x.__getitem__(y) <==> x[y]"""

        raise NotImplementedError("Subclass of FeatureStorage must implement `__getitem__(i: int)` method")

    def __len__(self) -> int:
        raise NotImplementedError("Subclass of FeatureStorage must implement `__len__` method")

    @abc.abstractmethod
    def __iter__(self) -> Iterator[FeatureVT]:
        """ Implement iter(self). """
        raise NotImplementedError("Subclass of FeatureStorage must implement `__iter__` method")

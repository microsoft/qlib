# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections.abc import MutableSequence, MutableMapping, Sequence
from typing import Iterable, overload, TypeVar, Tuple, List, Text, Iterator


# calendar value type
CalVT = TypeVar("CalVT")

# instrument value
InstVT = List[Tuple[CalVT, CalVT]]
# instrument key
InstKT = Text


FeatureVT = Tuple[int, float]


class CalendarStorage(MutableSequence):
    def __init__(self, uri: str):
        self._uri = uri

    def insert(self, index: int, o: CalVT) -> None:
        raise NotImplementedError("Subclass of CalendarStorage must implement `insert` method")

    @overload
    def __setitem__(self, i: int, o: CalVT) -> None:
        """x.__setitem__(i, o) <==> x[i] = o"""
        ...

    @overload
    def __setitem__(self, s: slice, o: Iterable[CalVT]) -> None:
        """x.__setitem__(s, o) <==> x[s] = o"""
        ...

    def __setitem__(self, i, o) -> None:
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


class InstrumentStorage(MutableMapping):
    def __init__(self, uri: str):
        self._uri = uri

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


class FeatureStorage(Sequence):
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
    def __getitem__(self, s: slice) -> Iterable[FeatureVT]:
        """x.__getitem__(slice(start: int, stop: int, step: int)) <==> x[start:stop:step]"""
        ...

    @overload
    def __getitem__(self, i: int) -> float:
        """x.__getitem__(y) <==> x[y]"""
        ...

    def __getitem__(self, i) -> float:
        """x.__getitem__(y) <==> x[y]"""
        raise NotImplementedError(
            "Subclass of FeatureStorage must implement `__getitem__(i: int)`/`__getitem__(s: slice)` method"
        )

    def __len__(self) -> int:
        raise NotImplementedError("Subclass of FeatureStorage must implement `__len__` method")

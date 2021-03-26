# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import struct
from pathlib import Path
from typing import Iterator, Iterable, Type, List, Tuple, Text, Union

from .storage import FeatureVT

import numpy as np
import pandas as pd
from . import CalendarStorage, InstrumentStorage, FeatureStorage


CalVT = Type[pd.Timestamp]
# instrument value
InstVT = List[Tuple[CalVT, CalVT]]
# instrument key
InstKT = Text


class FileCalendarStorage(CalendarStorage):
    def __init__(self, uri: str):
        super(FileCalendarStorage, self).__init__(uri=uri)
        with open(uri) as f:
            self._data = [pd.Timestamp(x.strip()) for x in f]

    def __getitem__(self, i: Union[int, slice]) -> Union[CalVT, Iterable[CalVT]]:
        if isinstance(i, (int, slice)):
            return self._data[i]
        else:
            raise TypeError(f"type(i) = {type(i)}")

    def __len__(self) -> int:
        return len(self._data)


class FileInstrumentStorage(InstrumentStorage):
    def __init__(self, uri: str):
        super(FileInstrumentStorage, self).__init__(uri=uri)
        self._data = self._load_data()

    def _load_data(self):
        _instruments = dict()
        df = pd.read_csv(
            self._uri,
            sep="\t",
            usecols=[0, 1, 2],
            names=["inst", "start_datetime", "end_datetime"],
            dtype={"inst": str},
            parse_dates=["start_datetime", "end_datetime"],
        )
        for row in df.itertuples(index=False):
            _instruments.setdefault(row[0], []).append((row[1], row[2]))
        return _instruments

    def __getitem__(self, k: InstKT) -> InstVT:
        return self._data[k]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[InstKT]:
        return self._data.__iter__()


class FileFeatureStorage(FeatureStorage):
    def __getitem__(self, i: Union[int, slice]) -> Union[FeatureVT, Iterable[FeatureVT]]:
        with open(self._uri, "rb") as fp:
            ref_start_index = int(np.frombuffer(fp.read(4), dtype="<f")[0])

            if isinstance(i, int):
                if ref_start_index > i:
                    raise IndexError(f"{i}: start index is {ref_start_index}")
                fp.seek(4 * (i - ref_start_index) + 4)
                return i, struct.unpack("f", fp.read(4))
            elif isinstance(i, slice):
                start_index = i.start
                end_index = i.stop - 1
                si = max(ref_start_index, start_index)
                if si > end_index:
                    return []
                fp.seek(4 * (si - ref_start_index) + 4)
                # read n bytes
                count = end_index - si + 1
                data = np.frombuffer(fp.read(4 * count), dtype="<f")
                return list(zip(range(si, si + len(data)), data))
            else:
                raise TypeError(f"type(i) = {type(i)}")

    def __len__(self) -> int:
        return Path(self._uri).stat().st_size // 4 - 1

    def __iter__(self):
        with open(self._uri, "rb") as fp:
            ref_start_index = int(np.frombuffer(fp.read(4), dtype="<f")[0])
            fp.seek(4)
            # read n bytes
            data = np.frombuffer(fp.read(), dtype="<f")
            for v in zip(range(ref_start_index, ref_start_index + len(data)), data):
                yield v

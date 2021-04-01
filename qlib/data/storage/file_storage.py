# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import struct
from pathlib import Path
from typing import Iterator, Iterable, Union, Dict, Mapping, Tuple

import numpy as np
import pandas as pd

from . import CalendarStorage, InstrumentStorage, FeatureStorage, CalVT, InstKT, InstVT


class FileCalendarStorage(CalendarStorage):
    def __init__(self, uri: str):
        super(FileCalendarStorage, self).__init__(uri)
        self._uri = Path(self._uri).expanduser().resolve()

    def _read_calendar(self, skip_rows: int = 0, n_rows: int = None) -> np.ndarray:
        if not self._uri.exists():
            self._write_calendar(values=[])
        with self._uri.open("rb") as fp:
            return np.loadtxt(fp, str, skiprows=skip_rows, max_rows=n_rows, encoding="utf-8")

    def _write_calendar(self, values: Iterable[CalVT], mode: str = "wb"):
        with self._uri.open(mode=mode) as fp:
            np.savetxt(fp, values, fmt="%s", encoding="utf-8")

    def extend(self, values: Iterable[CalVT]) -> None:
        self._write_calendar(values, mode="ab")

    def clear(self) -> None:
        self._write_calendar(values=[])

    def index(self, value: CalVT) -> int:
        calendar = self._read_calendar()
        return int(np.argwhere(calendar == value)[0])

    def insert(self, index: int, value: CalVT):
        calendar = self._read_calendar()
        calendar = np.insert(calendar, index, value)
        self._write_calendar(values=calendar)

    def remove(self, value: CalVT) -> None:
        index = self.index(value)
        calendar = self._read_calendar()
        calendar = np.delete(calendar, index)
        self._write_calendar(values=calendar)

    def __setitem__(self, i: Union[int, slice], values: Union[CalVT, Iterable[CalVT]]) -> None:
        calendar = self._read_calendar()
        calendar[i] = values
        self._write_calendar(values=calendar)

    def __delitem__(self, i: Union[int, slice]) -> None:
        calendar = self._read_calendar()
        calendar = np.delete(calendar, i)
        self._write_calendar(values=calendar)

    def __getitem__(self, i: Union[int, slice]) -> Union[CalVT, Iterable[CalVT]]:
        return self._read_calendar()[i]

    def __len__(self) -> int:
        return len(self._read_calendar())

    def __iter__(self):
        with self._uri.open("r") as fp:
            yield fp.readline()


class FileInstrumentStorage(InstrumentStorage):
    INSTRUMENT_SEP = "\t"
    INSTRUMENT_START_FIELD = "start_datetime"
    INSTRUMENT_END_FIELD = "end_datetime"
    SYMBOL_FIELD_NAME = "instrument"

    def __init__(self, uri: str):
        super(FileInstrumentStorage, self).__init__(uri=uri)
        self._uri = Path(self._uri).expanduser().resolve()

    def _read_instrument(self) -> Dict[InstKT, InstVT]:
        if not self._uri.exists():
            self._write_instrument()

        _instruments = dict()
        df = pd.read_csv(
            self._uri,
            sep="\t",
            usecols=[0, 1, 2],
            names=[self.SYMBOL_FIELD_NAME, self.INSTRUMENT_START_FIELD, self.INSTRUMENT_END_FIELD],
            dtype={self.SYMBOL_FIELD_NAME: str},
            parse_dates=[self.INSTRUMENT_START_FIELD, self.INSTRUMENT_END_FIELD],
        )
        for row in df.itertuples(index=False):
            _instruments.setdefault(row[0], []).append((row[1], row[2]))
        return _instruments

    def _write_instrument(self, data: Dict[InstKT, InstVT] = None) -> None:
        if not data:
            with self._uri.open("w") as _:
                pass
            return

        res = []
        for inst, v_list in data.items():
            _df = pd.DataFrame(v_list, columns=[self.INSTRUMENT_START_FIELD, self.INSTRUMENT_END_FIELD])
            _df[self.SYMBOL_FIELD_NAME] = inst
            res.append(_df)

        df = pd.concat(res, sort=False)
        df.loc[:, [self.SYMBOL_FIELD_NAME, self.INSTRUMENT_START_FIELD, self.INSTRUMENT_END_FIELD]].to_csv(
            self._uri, header=False, sep=self.INSTRUMENT_SEP, index=False
        )
        df.to_csv(self._uri, sep="\t", encoding="utf-8", header=False, index=False)

    def clear(self) -> None:
        self._write_instrument(data={})

    def __setitem__(self, k: InstKT, v: InstVT) -> None:
        inst = self._read_instrument()
        inst[k] = v
        self._write_instrument(inst)

    def __delitem__(self, k: InstKT) -> None:
        inst = self._read_instrument()
        del inst[k]
        self._write_instrument(inst)

    def __getitem__(self, k: InstKT) -> InstVT:
        return self._read_instrument()[k]

    def __len__(self) -> int:
        inst = self._read_instrument()
        return len(inst)

    def __iter__(self) -> Iterator[InstKT]:
        for _inst in self._read_instrument().keys():
            yield _inst

    def update(self, *args, **kwargs) -> None:

        if len(args) > 1:
            raise TypeError(f"update expected at most 1 arguments, got {len(args)}")
        inst = self._read_instrument()
        if args:
            other = args[0]
            if isinstance(other, Mapping):
                for key in other:
                    inst[key] = other[key]
            elif hasattr(other, "keys"):
                for key in other.keys():
                    inst[key] = other[key]
            else:
                for key, value in other:
                    inst[key] = value
        for key, value in kwargs.items():
            inst[key] = value

        self._write_instrument(inst)


class FileFeatureStorage(FeatureStorage):
    def __init__(self, uri: str):
        super(FileFeatureStorage, self).__init__(uri=uri)
        self._uri = Path(self._uri)

    def clear(self):
        with self._uri.open("wb") as _:
            pass

    def extend(self, series: pd.Series) -> None:
        extend_start_index = self[0][0] + len(self) if self._uri.exists() else series.index[0]
        series = series.reindex(pd.RangeIndex(extend_start_index, series.index[-1] + 1))
        with self._uri.open("ab") as fp:
            np.array(series.values).astype("<f").tofile(fp)

    def rebase(self, series: pd.Series) -> None:
        origin_series = self[:]
        series = series.append(origin_series.loc[origin_series.index > series.index[-1]])
        series = series.reindex(pd.RangeIndex(series.index[0], series.index[-1]))
        with self._uri.open("wb") as fp:
            np.array(series.values).astype("<f").tofile(fp)

    def __getitem__(self, i: Union[int, slice]) -> Union[Tuple[int, float], pd.Series]:
        if not self._uri.exists():
            if isinstance(i, int):
                return None, None
            elif isinstance(i, slice):
                return pd.Series()
            else:
                raise TypeError(f"type(i) = {type(i)}")

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
                    return pd.Series()
                fp.seek(4 * (si - ref_start_index) + 4)
                # read n bytes
                count = end_index - si + 1
                data = np.frombuffer(fp.read(4 * count), dtype="<f")
                return pd.Series(data, index=pd.RangeIndex(si, si + len(data)))
            else:
                raise TypeError(f"type(i) = {type(i)}")

    def __len__(self) -> int:
        return self._uri.stat().st_size // 4 - 1 if self._uri.exists() else 0

    def __iter__(self):
        if not self._uri.exists():
            return
        with open(self._uri, "rb") as fp:
            ref_start_index = int(np.frombuffer(fp.read(4), dtype="<f")[0])
            fp.seek(4)
            while True:
                v = fp.read(4)
                if v:
                    yield ref_start_index, struct.unpack("f", v)
                    ref_start_index += 1
                else:
                    break

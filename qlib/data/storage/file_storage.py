# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import struct
from pathlib import Path
from typing import Iterable, Union, Dict, Mapping, Tuple, List

import numpy as np
import pandas as pd

from qlib.log import get_module_logger
from qlib.data.storage import CalendarStorage, InstrumentStorage, FeatureStorage, CalVT, InstKT, InstVT

logger = get_module_logger("file_storage")


class FileStorageMixin:
    @property
    def uri(self) -> Path:
        _provider_uri = self.kwargs.get("provider_uri", None)
        if _provider_uri is None:
            raise ValueError(
                f"The `provider_uri` parameter is not found in {self.__class__.__name__}, "
                f'please specify `provider_uri` in the "provider\'s backend"'
            )
        return Path(_provider_uri).expanduser().joinpath(f"{self.storage_name}s", self.file_name)

    def check(self):
        """check self.uri

        Raises
        -------
        ValueError
        """
        if not self.uri.exists():
            raise ValueError(f"{self.storage_name} not exists: {self.uri}")


class FileCalendarStorage(FileStorageMixin, CalendarStorage):
    def __init__(self, freq: str, future: bool, **kwargs):
        super(FileCalendarStorage, self).__init__(freq, future, **kwargs)
        self.file_name = f"{freq}_future.txt" if future else f"{freq}.txt".lower()

    def _read_calendar(self, skip_rows: int = 0, n_rows: int = None) -> List[CalVT]:
        if not self.uri.exists():
            self._write_calendar(values=[])
        with self.uri.open("rb") as fp:
            return [
                str(x)
                for x in np.loadtxt(fp, str, skiprows=skip_rows, max_rows=n_rows, delimiter="\n", encoding="utf-8")
            ]

    def _write_calendar(self, values: Iterable[CalVT], mode: str = "wb"):
        with self.uri.open(mode=mode) as fp:
            np.savetxt(fp, values, fmt="%s", encoding="utf-8")

    @property
    def data(self) -> List[CalVT]:
        self.check()
        return self._read_calendar()

    def extend(self, values: Iterable[CalVT]) -> None:
        self._write_calendar(values, mode="ab")

    def clear(self) -> None:
        self._write_calendar(values=[])

    def index(self, value: CalVT) -> int:
        self.check()
        calendar = self._read_calendar()
        return int(np.argwhere(calendar == value)[0])

    def insert(self, index: int, value: CalVT):
        calendar = self._read_calendar()
        calendar = np.insert(calendar, index, value)
        self._write_calendar(values=calendar)

    def remove(self, value: CalVT) -> None:
        self.check()
        index = self.index(value)
        calendar = self._read_calendar()
        calendar = np.delete(calendar, index)
        self._write_calendar(values=calendar)

    def __setitem__(self, i: Union[int, slice], values: Union[CalVT, Iterable[CalVT]]) -> None:
        calendar = self._read_calendar()
        calendar[i] = values
        self._write_calendar(values=calendar)

    def __delitem__(self, i: Union[int, slice]) -> None:
        self.check()
        calendar = self._read_calendar()
        calendar = np.delete(calendar, i)
        self._write_calendar(values=calendar)

    def __getitem__(self, i: Union[int, slice]) -> Union[CalVT, List[CalVT]]:
        self.check()
        return self._read_calendar()[i]

    def __len__(self) -> int:
        return len(self.data)


class FileInstrumentStorage(FileStorageMixin, InstrumentStorage):

    INSTRUMENT_SEP = "\t"
    INSTRUMENT_START_FIELD = "start_datetime"
    INSTRUMENT_END_FIELD = "end_datetime"
    SYMBOL_FIELD_NAME = "instrument"

    def __init__(self, market: str, **kwargs):
        super(FileInstrumentStorage, self).__init__(market, **kwargs)
        self.file_name = f"{market.lower()}.txt"

    def _read_instrument(self) -> Dict[InstKT, InstVT]:
        if not self.uri.exists():
            self._write_instrument()

        _instruments = dict()
        df = pd.read_csv(
            self.uri,
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
            with self.uri.open("w") as _:
                pass
            return

        res = []
        for inst, v_list in data.items():
            _df = pd.DataFrame(v_list, columns=[self.INSTRUMENT_START_FIELD, self.INSTRUMENT_END_FIELD])
            _df[self.SYMBOL_FIELD_NAME] = inst
            res.append(_df)

        df = pd.concat(res, sort=False)
        df.loc[:, [self.SYMBOL_FIELD_NAME, self.INSTRUMENT_START_FIELD, self.INSTRUMENT_END_FIELD]].to_csv(
            self.uri, header=False, sep=self.INSTRUMENT_SEP, index=False
        )
        df.to_csv(self.uri, sep="\t", encoding="utf-8", header=False, index=False)

    def clear(self) -> None:
        self._write_instrument(data={})

    @property
    def data(self) -> Dict[InstKT, InstVT]:
        self.check()
        return self._read_instrument()

    def __setitem__(self, k: InstKT, v: InstVT) -> None:
        inst = self._read_instrument()
        inst[k] = v
        self._write_instrument(inst)

    def __delitem__(self, k: InstKT) -> None:
        self.check()
        inst = self._read_instrument()
        del inst[k]
        self._write_instrument(inst)

    def __getitem__(self, k: InstKT) -> InstVT:
        self.check()
        return self._read_instrument()[k]

    def update(self, *args, **kwargs) -> None:

        if len(args) > 1:
            raise TypeError(f"update expected at most 1 arguments, got {len(args)}")
        inst = self._read_instrument()
        if args:
            other = args[0]  # type: dict
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

    def __len__(self) -> int:
        return len(self.data)


class FileFeatureStorage(FileStorageMixin, FeatureStorage):
    def __init__(self, instrument: str, field: str, freq: str, **kwargs):
        super(FileFeatureStorage, self).__init__(instrument, field, freq, **kwargs)
        self.file_name = f"{instrument.lower()}/{field.lower()}.{freq.lower()}.bin"

    def clear(self):
        with self.uri.open("wb") as _:
            pass

    @property
    def data(self) -> pd.Series:
        return self[:]

    def write(self, data_array: Union[List, np.ndarray], index: int = None) -> None:
        if len(data_array) == 0:
            logger.info(
                "len(data_array) == 0, write"
                "if you need to clear the FeatureStorage, please execute: FeatureStorage.clear"
            )
            return
        if not self.uri.exists():
            # write
            index = 0 if index is None else index
            with self.uri.open("wb") as fp:
                np.hstack([index, data_array]).astype("<f").tofile(fp)
        else:
            if index is None or index > self.end_index:
                # append
                index = 0 if index is None else index
                with self.uri.open("ab+") as fp:
                    np.hstack([[np.nan] * (index - self.end_index - 1), data_array]).astype("<f").tofile(fp)
            else:
                # rewrite
                with self.uri.open("rb+") as fp:
                    _old_data = np.fromfile(fp, dtype="<f")
                    _old_index = _old_data[0]
                    _old_df = pd.DataFrame(
                        _old_data[1:], index=range(_old_index, _old_index + len(_old_data) - 1), columns=["old"]
                    )
                    fp.seek(0)
                    _new_df = pd.DataFrame(data_array, index=range(index, index + len(data_array)), columns=["new"])
                    _df = pd.concat([_old_df, _new_df], sort=False, axis=1)
                    _df = _df.reindex(range(_df.index.min(), _df.index.max() + 1))
                    _df["new"].fillna(_df["old"]).values.astype("<f").tofile(fp)

    @property
    def start_index(self) -> Union[int, None]:
        if not self.uri.exists():
            return None
        with self.uri.open("rb") as fp:
            index = int(np.frombuffer(fp.read(4), dtype="<f")[0])
        return index

    @property
    def end_index(self) -> Union[int, None]:
        if not self.uri.exists():
            return None
        # The next  data appending index point will be  `end_index + 1`
        return self.start_index + len(self) - 1

    def __getitem__(self, i: Union[int, slice]) -> Union[Tuple[int, float], pd.Series]:
        if not self.uri.exists():
            if isinstance(i, int):
                return None, None
            elif isinstance(i, slice):
                return pd.Series(dtype=np.float32)
            else:
                raise TypeError(f"type(i) = {type(i)}")

        storage_start_index = self.start_index
        storage_end_index = self.end_index
        with self.uri.open("rb") as fp:
            if isinstance(i, int):

                if storage_start_index > i:
                    raise IndexError(f"{i}: start index is {storage_start_index}")
                fp.seek(4 * (i - storage_start_index) + 4)
                return i, struct.unpack("f", fp.read(4))[0]
            elif isinstance(i, slice):
                start_index = storage_start_index if i.start is None else i.start
                end_index = storage_end_index if i.stop is None else i.stop - 1
                si = max(start_index, storage_start_index)
                if si > end_index:
                    return pd.Series(dtype=np.float32)
                fp.seek(4 * (si - storage_start_index) + 4)
                # read n bytes
                count = end_index - si + 1
                data = np.frombuffer(fp.read(4 * count), dtype="<f")
                return pd.Series(data, index=pd.RangeIndex(si, si + len(data)))
            else:
                raise TypeError(f"type(i) = {type(i)}")

    def __len__(self) -> int:
        self.check()
        return self.uri.stat().st_size // 4 - 1

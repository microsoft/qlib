# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import struct
from pathlib import Path
from typing import Iterable, Union, Dict, Mapping, Tuple, List

import numpy as np
import pandas as pd
from qlib.data.storage.storage import PITStorage

from qlib.utils.time import Freq
from qlib.utils.resam import resam_calendar
from qlib.config import C
from qlib.data.cache import H
from qlib.log import get_module_logger
from qlib.data.storage import (
    CalendarStorage,
    InstrumentStorage,
    FeatureStorage,
    CalVT,
    InstKT,
    InstVT,
)

logger = get_module_logger("file_storage")


class FileStorageMixin:
    """FileStorageMixin, applicable to FileXXXStorage
    Subclasses need to have provider_uri, freq, storage_name, file_name attributes

    """

    # NOTE: provider_uri priority:
    #   1. self._provider_uri : if provider_uri is provided.
    #   2. provider_uri in qlib.config.C

    @property
    def provider_uri(self):
        return C["provider_uri"] if getattr(self, "_provider_uri", None) is None else self._provider_uri

    @property
    def dpm(self):
        return (
            C.dpm
            if getattr(self, "_provider_uri", None) is None
            else C.DataPathManager(self._provider_uri, C.mount_path)
        )

    @property
    def support_freq(self) -> List[str]:
        _v = "_support_freq"
        if hasattr(self, _v):
            return getattr(self, _v)
        if len(self.provider_uri) == 1 and C.DEFAULT_FREQ in self.provider_uri:
            freq_l = filter(
                lambda _freq: not _freq.endswith("_future"),
                map(
                    lambda x: x.stem,
                    self.dpm.get_data_uri(C.DEFAULT_FREQ).joinpath("calendars").glob("*.txt"),
                ),
            )
        else:
            freq_l = self.provider_uri.keys()
        freq_l = [Freq(freq) for freq in freq_l]
        setattr(self, _v, freq_l)
        return freq_l

    @property
    def uri(self) -> Path:
        if self.freq not in self.support_freq:
            raise ValueError(f"{self.storage_name}: {self.provider_uri} does not contain data for {self.freq}")
        return self.dpm.get_data_uri(self.freq).joinpath(f"{self.storage_name}s", self.file_name)

    def check(self):
        """check self.uri

        Raises
        -------
        ValueError
        """
        if not self.uri.exists():
            raise ValueError(f"{self.storage_name} not exists: {self.uri}")


class FileCalendarStorage(FileStorageMixin, CalendarStorage):
    def __init__(self, freq: str, future: bool, provider_uri: dict = None, **kwargs):
        super(FileCalendarStorage, self).__init__(freq, future, **kwargs)
        self.future = future
        self._provider_uri = None if provider_uri is None else C.DataPathManager.format_provider_uri(provider_uri)
        self.enable_read_cache = True  # TODO: make it configurable
        self.region = C["region"]

    @property
    def file_name(self) -> str:
        return f"{self._freq_file}_future.txt" if self.future else f"{self._freq_file}.txt".lower()

    @property
    def _freq_file(self) -> str:
        """the freq to read from file"""
        if not hasattr(self, "_freq_file_cache"):
            freq = Freq(self.freq)
            if freq not in self.support_freq:
                # NOTE: uri
                #   1. If `uri` does not exist
                #       - Get the `min_uri` of the closest `freq` under the same "directory" as the `uri`
                #       - Read data from `min_uri` and resample to `freq`

                freq = Freq.get_recent_freq(freq, self.support_freq)
                if freq is None:
                    raise ValueError(f"can't find a freq from {self.support_freq} that can resample to {self.freq}!")
            self._freq_file_cache = freq
        return self._freq_file_cache

    def _read_calendar(self) -> List[CalVT]:
        # NOTE:
        # if we want to accelerate partial reading calendar
        # we can add parameters like `skip_rows: int = 0, n_rows: int = None` to the interface.
        # Currently, it is not supported for the txt-based calendar

        if not self.uri.exists():
            self._write_calendar(values=[])

        with self.uri.open("r") as fp:
            res = []
            for line in fp.readlines():
                line = line.strip()
                if len(line) > 0:
                    res.append(line)
            return res

    def _write_calendar(self, values: Iterable[CalVT], mode: str = "wb"):
        with self.uri.open(mode=mode) as fp:
            np.savetxt(fp, values, fmt="%s", encoding="utf-8")

    @property
    def uri(self) -> Path:
        return self.dpm.get_data_uri(self._freq_file).joinpath(f"{self.storage_name}s", self.file_name)

    @property
    def data(self) -> List[CalVT]:
        self.check()
        # If cache is enabled, then return cache directly
        if self.enable_read_cache:
            key = "orig_file" + str(self.uri)
            if key not in H["c"]:
                H["c"][key] = self._read_calendar()
            _calendar = H["c"][key]
        else:
            _calendar = self._read_calendar()
        if Freq(self._freq_file) != Freq(self.freq):
            _calendar = resam_calendar(
                np.array(list(map(pd.Timestamp, _calendar))),
                self._freq_file,
                self.freq,
                self.region,
            )
        return _calendar

    def _get_storage_freq(self) -> List[str]:
        return sorted(set(map(lambda x: x.stem.split("_")[0], self.uri.parent.glob("*.txt"))))

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

    def __init__(self, market: str, freq: str, provider_uri: dict = None, **kwargs):
        super(FileInstrumentStorage, self).__init__(market, freq, **kwargs)
        self._provider_uri = None if provider_uri is None else C.DataPathManager.format_provider_uri(provider_uri)
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
    def __init__(self, instrument: str, field: str, freq: str, provider_uri: dict = None, **kwargs):
        super(FileFeatureStorage, self).__init__(instrument, field, freq, **kwargs)
        self._provider_uri = None if provider_uri is None else C.DataPathManager.format_provider_uri(provider_uri)
        self.file_name = f"{instrument.lower()}/{field.lower()}.{freq.lower()}.bin"
        self._start_index = None

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
        self._start_index = None
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
                        _old_data[1:],
                        index=range(_old_index, _old_index + len(_old_data) - 1),
                        columns=["old"],
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
        if self._start_index is None:
            with self.uri.open("rb") as fp:
                self._start_index = int(np.frombuffer(fp.read(4), dtype="<f")[0])
        return self._start_index

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


class FilePITStorage(FileStorageMixin, PITStorage):
    """PIT data is a special case of Feature data, it looks like

                date  period     value       _next
            0  20070428  200701  0.090219  4294967295
            1  20070817  200702  0.139330  4294967295
            2  20071023  200703  0.245863  4294967295
            3  20080301  200704  0.347900          80
            4  20080313  200704  0.395989  4294967295

    It is sorted by [date, period].

    next field currently is not used. just for forward compatible.
    """

    # NOTE:
    # PIT data should have two files, one is the index file, the other is the data file.

    # pesudo code:
    # date_index = calendar.index(date)
    # data_start_index, data_end_index = index_file[date_index]
    # data = data_file[data_start_index:data_end_index]

    # the index file is like feature's data file, but given a start index in index file, it will return the first and the last observe index of the data file.
    # the data file has tree columns, the first column is observe date, the second column is financial period, the third column is the value.

    # so given start and end date, we can get the start_index and end_index from calendar.
    # use it to read two line from index file, then we can get the start and end index of the data file.

    # but consider this implementation, we will create a index file which will have 50 times lines than the data file. Is it a good idea?
    # if we just create a index file the same line with data file, we have to read the whole index file for any time slice search, so why not read whole data file?

    def __init__(self, instrument: str, field: str, freq: str = "day", provider_uri: dict = None, **kwargs):
        super(FilePITStorage, self).__init__(instrument, field, freq, **kwargs)

        if not field.endswith("_q") and not field.endswith("_a"):
            raise ValueError("period field must ends with '_q' or '_a'")
        self.quarterly = field.endswith("_q")

        self._provider_uri = None if provider_uri is None else C.DataPathManager.format_provider_uri(provider_uri)
        self.file_name = f"{instrument.lower()}/{field.lower()}.data"
        self.uri.parent.mkdir(parents=True, exist_ok=True)
        self.raw_dtype = [
            ("date", C.pit_record_type["date"]),
            ("period", C.pit_record_type["period"]),
            ("value", C.pit_record_type["value"]),
            ("_next", C.pit_record_type["index"]),  # not used in current implementation
        ]
        self.dtypes = np.dtype(self.raw_dtype)
        self.itemsize = self.dtypes.itemsize
        self.dtype_string = "".join([i[1] for i in self.raw_dtype])
        self.columns = [i[0] for i in self.raw_dtype]

    @property
    def uri(self) -> Path:
        if self.freq not in self.support_freq:
            raise ValueError(f"{self.storage_name}: {self.provider_uri} does not contain data for {self.freq}")
        return self.dpm.get_data_uri(self.freq).joinpath(f"{self.storage_name}", self.file_name)

    def clear(self):
        with self.uri.open("wb") as _:
            pass

    @property
    def data(self) -> pd.DataFrame:
        return self[:]

    def update(self, data_array: np.ndarray) -> None:
        """update data to storage, replace current data from start_date to end_date with given data_array

        Args:
            data_array: Structured arrays contains date, period, value and next. same with self.raw_dtype
        """
        if not self.uri.exists() or len(self) == 0:
            # write
            index = 0
            self.write(data_array, index)
        else:
            # sort it
            data_array = np.sort(data_array, order=["date", "period"])
            # get index
            update_start_date = data_array[0][0]
            update_end_date = data_array[-1][0]
            current_data = self.np_data()
            index = (current_data["date"] >= update_start_date).argmax()
            end_index = (current_data["date"] > update_end_date).argmax()
            new_data = np.concatenate([data_array, current_data[end_index:]])
            self.write(new_data, index)

    def write(self, data_array: np.ndarray, index: int = None) -> None:
        """write data to storage at specific index

        Args:
            data_array: Structured arrays contains date, period, value and next
            index: target index to start writing. Defaults to None.
        """

        if len(data_array) == 0:
            logger.info(
                "len(data_array) == 0, write"
                "if you need to clear the FeatureStorage, please execute: FeatureStorage.clear"
            )
            return
        # check data_array dtype
        if data_array.dtype != self.dtypes:
            raise ValueError(f"data_array.dtype = {data_array.dtype}, self.dtypes = {self.dtypes}")

        # sort data_array with first 2 columns
        data_array = np.sort(data_array, order=["date", "period"])

        if not self.uri.exists():
            # write
            index = 0 if index is None else index
            with self.uri.open("wb") as fp:
                data_array.tofile(fp)
        else:
            if index is None or index > self.end_index:
                index = self.end_index + 1
            with self.uri.open("rb+") as fp:
                fp.seek(index * self.itemsize)
                data_array.tofile(fp)

    @property
    def start_index(self) -> Union[int, None]:
        return 0

    @property
    def end_index(self) -> Union[int, None]:
        if not self.uri.exists():
            return None
        # The next  data appending index point will be  `end_index + 1`
        return self.start_index + len(self) - 1

    def np_data(self, i: Union[int, slice] = None) -> np.ndarray:
        """return numpy structured array

        Args:
            i: index or slice. Defaults to None.

        Returns:
            np.ndarray
        """
        if not self.uri.exists():
            if isinstance(i, int):
                return None, None
            elif isinstance(i, slice):
                return np.array(dtype=self.dtypes)
            else:
                raise TypeError(f"type(i) = {type(i)}")

        if i is None:
            i = slice(None, None)
        storage_start_index = self.start_index
        storage_end_index = self.end_index
        with self.uri.open("rb") as fp:
            if isinstance(i, int):
                if storage_start_index > i:
                    raise IndexError(f"{i}: start index is {storage_start_index}")
                fp.seek(i * self.itemsize)
                return np.array([struct.unpack(self.dtype_string, fp.read(self.itemsize))], dtype=self.dtypes)
            elif isinstance(i, slice):
                start_index = storage_start_index if i.start is None else i.start
                end_index = storage_end_index if i.stop is None else i.stop - 1
                si = max(start_index, storage_start_index)
                if si > end_index:
                    return np.array(dtype=self.dtypes)
                fp.seek(start_index * self.itemsize)
                # read n bytes
                count = end_index - si + 1
                data = np.frombuffer(fp.read(self.itemsize * count), dtype=self.dtypes)
                return data
            else:
                raise TypeError(f"type(i) = {type(i)}")

    def __getitem__(self, i: Union[int, slice]) -> Union[Tuple[int, float], pd.DataFrame]:
        if isinstance(i, int):
            return pd.Series(self.np_data(i), index=self.columns, name=i)
        elif isinstance(i, slice):
            data = self.np_data(i)
            si = self.start_index if i.start is None else i.start
            if si < 0:
                si = len(self) + si
            return pd.DataFrame(data, index=pd.RangeIndex(si, si + len(data)), columns=self.columns)
        else:
            raise TypeError(f"type(i) = {type(i)}")

    def __len__(self) -> int:
        self.check()
        return self.uri.stat().st_size // self.itemsize

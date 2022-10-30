from typing import Union, List, Dict, Tuple
import unittest
import pandas as pd
import numpy as np
import io

from .data import GetData
from .. import init
from ..constant import REG_CN, REG_TW
from qlib.data.filter import NameDFilter
from qlib.data import D
from qlib.data.data import Cal, DatasetD
from qlib.data.storage import CalendarStorage, InstrumentStorage, FeatureStorage, CalVT, InstKT, InstVT


class TestAutoData(unittest.TestCase):

    _setup_kwargs = {}
    provider_uri = "~/.qlib/qlib_data/cn_data_simple"  # target_dir
    provider_uri_1day = "~/.qlib/qlib_data/cn_data"  # target_dir
    provider_uri_1min = "~/.qlib/qlib_data/cn_data_1min"

    @classmethod
    def setUpClass(cls, enable_1d_type="simple", enable_1min=False) -> None:
        # use default data

        if enable_1d_type == "simple":
            provider_uri_day = cls.provider_uri
            name_day = "qlib_data_simple"
        elif enable_1d_type == "full":
            provider_uri_day = cls.provider_uri_1day
            name_day = "qlib_data"
        else:
            raise NotImplementedError(f"This type of input is not supported")

        GetData().qlib_data(
            name=name_day,
            region=REG_CN,
            interval="1d",
            target_dir=provider_uri_day,
            delete_old=False,
            exists_skip=True,
        )

        if enable_1min:
            GetData().qlib_data(
                name="qlib_data",
                region=REG_CN,
                interval="1min",
                target_dir=cls.provider_uri_1min,
                delete_old=False,
                exists_skip=True,
            )

        provider_uri_map = {"1min": cls.provider_uri_1min, "day": provider_uri_day}
        init(
            provider_uri=provider_uri_map,
            region=REG_CN,
            expression_cache=None,
            dataset_cache=None,
            **cls._setup_kwargs,
        )


class TestOperatorData(TestAutoData):
    @classmethod
    def setUpClass(cls, enable_1d_type="simple", enable_1min=False) -> None:
        # use default data
        super().setUpClass(enable_1d_type, enable_1min)
        nameDFilter = NameDFilter(name_rule_re="SH600110")
        instruments = D.instruments("csi300", filter_pipe=[nameDFilter])
        start_time = "2005-01-04"
        end_time = "2005-12-31"
        freq = "day"

        instruments_d = DatasetD.get_instruments_d(instruments, freq)
        cls.instruments_d = instruments_d
        cal = Cal.calendar(start_time, end_time, freq)
        cls.cal = cal
        cls.start_time = cal[0]
        cls.end_time = cal[-1]
        cls.inst = list(instruments_d.keys())[0]
        cls.spans = list(instruments_d.values())[0]


MOCK_DATA = """
id,symbol,datetime,interval,volume,open,high,low,close
20275,0050,2022-01-03 00:00:00,day,6761.0,146.0,147.35,146.0,146.4
20276,0050,2022-01-04 00:00:00,day,9608.0,147.7,149.6,147.7,149.6
20277,0050,2022-01-05 00:00:00,day,11387.0,150.1,150.55,149.1,149.3
20278,0050,2022-01-06 00:00:00,day,8611.0,148.3,148.75,147.0,147.9
20279,0050,2022-01-07 00:00:00,day,6954.0,148.3,149.0,146.5,146.6
20280,0050,2022-01-10 00:00:00,day,15684.0,146.0,147.8,145.4,147.55
20281,0050,2022-01-11 00:00:00,day,17741.0,147.6,148.5,146.7,148.3
20282,0050,2022-01-12 00:00:00,day,10134.0,149.35,149.6,148.7,149.55
20283,0050,2022-01-13 00:00:00,day,7431.0,149.55,150.45,149.55,150.3
20284,0050,2022-01-14 00:00:00,day,10091.0,150.8,151.2,149.05,150.3
20285,0050,2022-01-17 00:00:00,day,6899.0,151.1,152.4,151.1,152.0
20286,0050,2022-01-18 00:00:00,day,14360.0,152.2,152.25,150.15,150.3
20287,0050,2022-01-19 00:00:00,day,14654.0,149.0,149.65,148.25,148.5
20288,0050,2022-01-20 00:00:00,day,16201.0,148.5,149.2,147.6,149.1
20289,0050,2022-01-21 00:00:00,day,29848.0,143.9,143.95,142.3,142.65
20290,0050,2022-01-24 00:00:00,day,13143.0,142.1,144.0,141.7,144.0
20291,0050,2022-01-25 00:00:00,day,23982.0,142.55,142.55,141.25,141.65
20292,0050,2022-01-26 00:00:00,day,17729.0,141.15,142.2,141.05,141.55
8547,1101,2021-12-01 00:00:00,day,16119.0,46.0,46.85,46.0,46.6
8548,1101,2021-12-02 00:00:00,day,14521.0,46.6,46.7,46.3,46.3
8549,1101,2021-12-03 00:00:00,day,14357.0,46.55,46.85,46.4,46.4
8550,1101,2021-12-06 00:00:00,day,15115.0,46.45,47.35,46.4,47.3
8551,1101,2021-12-07 00:00:00,day,13117.0,47.35,47.55,46.9,47.55
8552,1101,2021-12-08 00:00:00,day,10329.0,47.75,47.8,47.5,47.7
8553,1101,2021-12-09 00:00:00,day,9300.0,47.8,47.85,47.1,47.4
8554,1101,2021-12-10 00:00:00,day,9919.0,47.4,47.6,47.1,47.3
8555,1101,2021-12-13 00:00:00,day,7784.0,47.3,47.75,47.1,47.1
8556,1101,2021-12-14 00:00:00,day,9373.0,47.05,47.2,46.95,47.0
8557,1101,2021-12-15 00:00:00,day,11189.0,47.0,47.3,46.8,46.95
8558,1101,2021-12-16 00:00:00,day,7516.0,47.0,47.15,46.8,46.9
8559,1101,2021-12-17 00:00:00,day,18502.0,46.95,47.6,46.9,47.45
8560,1101,2021-12-20 00:00:00,day,11309.0,47.45,47.5,47.1,47.4
8561,1101,2021-12-21 00:00:00,day,5666.0,47.4,47.45,47.1,47.25
8562,1101,2021-12-22 00:00:00,day,5460.0,47.4,47.45,47.2,47.4
8563,1101,2021-12-23 00:00:00,day,9371.0,47.3,47.7,47.3,47.7
8564,1101,2021-12-24 00:00:00,day,5980.0,47.75,47.95,47.75,47.9
8565,1101,2021-12-27 00:00:00,day,5709.0,47.9,48.1,47.9,48.1
8566,1101,2021-12-28 00:00:00,day,7777.0,48.1,48.15,47.95,48.15
8567,1101,2021-12-29 00:00:00,day,5309.0,48.15,48.25,48.05,48.15
8568,1101,2021-12-30 00:00:00,day,4616.0,48.15,48.2,48.0,48.0
8569,1101,2022-01-03 00:00:00,day,12350.0,48.05,48.15,47.35,47.45
8570,1101,2022-01-04 00:00:00,day,11439.0,47.5,47.6,47.0,47.3
8571,1101,2022-01-05 00:00:00,day,9692.0,47.1,47.3,47.0,47.15
8572,1101,2022-01-06 00:00:00,day,12361.0,47.3,47.6,47.15,47.6
8573,1101,2022-01-07 00:00:00,day,10921.0,47.6,47.65,47.2,47.45
8574,1101,2022-01-10 00:00:00,day,11925.0,47.45,47.5,47.0,47.3
8575,1101,2022-01-11 00:00:00,day,11047.0,47.1,47.5,47.1,47.5
8576,1101,2022-01-12 00:00:00,day,10817.0,47.5,47.5,47.1,47.5
8577,1101,2022-01-13 00:00:00,day,13849.0,47.5,47.95,47.4,47.95
8578,1101,2022-01-14 00:00:00,day,9460.0,47.85,47.85,47.45,47.6
8579,1101,2022-01-17 00:00:00,day,9057.0,47.55,47.7,47.35,47.6
8580,1101,2022-01-18 00:00:00,day,8089.0,47.6,47.75,47.45,47.75
8581,1101,2022-01-19 00:00:00,day,5110.0,47.6,47.7,47.5,47.6
8582,1101,2022-01-20 00:00:00,day,6327.0,47.55,47.7,47.45,47.5
8583,1101,2022-01-21 00:00:00,day,9470.0,47.5,47.65,47.15,47.4
8584,1101,2022-01-24 00:00:00,day,5475.0,47.1,47.3,47.0,47.15
8585,1101,2022-01-25 00:00:00,day,16153.0,47.0,47.05,46.6,46.8
8586,1101,2022-01-26 00:00:00,day,7772.0,46.7,47.0,46.55,46.85
8587,1101,2022-02-07 00:00:00,day,17031.0,46.55,47.1,46.0,47.1
8588,1101,2022-02-08 00:00:00,day,9741.0,47.1,47.25,46.9,46.95
8589,1101,2022-02-09 00:00:00,day,7968.0,46.95,47.3,46.9,47.3
8590,1101,2022-02-10 00:00:00,day,7479.0,47.15,47.55,47.05,47.55
8591,1101,2022-02-11 00:00:00,day,6841.0,47.3,47.55,47.15,47.55
8592,1101,2022-02-14 00:00:00,day,9136.0,47.2,47.3,46.95,47.15
8593,1101,2022-02-15 00:00:00,day,5444.0,47.05,47.1,46.8,47.0
8594,1101,2022-02-16 00:00:00,day,8751.0,47.0,47.15,47.0,47.0
8595,1101,2022-02-17 00:00:00,day,10662.0,47.15,47.55,47.1,47.45
8596,1101,2022-02-18 00:00:00,day,8781.0,47.25,47.55,47.2,47.45
8597,1101,2022-02-21 00:00:00,day,8201.0,47.35,47.75,47.15,47.6
8598,1101,2022-02-22 00:00:00,day,10655.0,47.4,47.7,47.1,47.7
8599,1101,2022-02-23 00:00:00,day,8040.0,47.7,47.85,47.45,47.65
8600,1101,2022-02-24 00:00:00,day,13124.0,47.5,47.5,47.1,47.3
8601,1101,2022-02-25 00:00:00,day,14556.0,47.2,47.5,46.9,47.35
"""

MOCK_DF = pd.read_csv(io.StringIO(MOCK_DATA), header=0, dtype={"symbol": str})


class MockStorageBase:
    def __init__(self, **kwargs):
        self.df = MOCK_DF


class MockCalendarStorage(MockStorageBase, CalendarStorage):
    def __init__(self, **kwargs):
        super().__init__()
        self._data = sorted(self.df["datetime"].unique())

    @property
    def data(self) -> List[CalVT]:
        return self._data

    def __getitem__(self, i: Union[int, slice]) -> Union[CalVT, List[CalVT]]:
        return self.data[i]

    def __len__(self) -> int:
        return len(self.data)


class MockInstrumentStorage(MockStorageBase, InstrumentStorage):
    def __init__(self, **kwargs):
        super().__init__()
        instruments = {}
        for symbol, group in self.df.groupby(by="symbol"):
            start = group["datetime"].iloc[0]
            end = group["datetime"].iloc[-1]
            instruments[symbol] = [(start, end)]
        self._data = instruments

    @property
    def data(self) -> Dict[InstKT, InstVT]:
        return self._data

    def __getitem__(self, k: InstKT) -> InstVT:
        return self.data[k]

    def __len__(self) -> int:
        return len(self.data)


class MockFeatureStorage(MockStorageBase, FeatureStorage):
    def __init__(self, instrument: str, field: str, freq: str, db_region: str = None, **kwargs):  # type: ignore
        super().__init__(instrument=instrument, field=field, freq=freq, db_region=db_region, **kwargs)
        self.field = field
        calendar = sorted(self.df["datetime"].unique())
        df_calendar = pd.DataFrame(calendar, columns=["datetime"]).set_index("datetime")
        df = self.df[self.df["symbol"] == instrument]
        data_dt_field = "datetime"
        cal_df = df_calendar[
            (df_calendar.index >= df[data_dt_field].min()) & (df_calendar.index <= df[data_dt_field].max())
        ]
        df = df.set_index(data_dt_field)
        df_data = df.reindex(cal_df.index)
        date_index = df_calendar.index.get_loc(df_data.index.min())  # type: ignore
        df_data.reset_index(inplace=True)
        df_data.index += date_index
        self._data = df_data

    @property
    def data(self) -> pd.Series:
        return self._data[self.field]

    @property
    def start_index(self) -> Union[int, None]:
        if self._data.empty:
            return None
        return self._data.index[0]

    @property
    def end_index(self) -> Union[int, None]:
        if self._data.empty:
            return None
        # The next  data appending index point will be  `end_index + 1`
        return self._data.index[-1]

    def __getitem__(self, i: Union[int, slice]) -> Union[Tuple[int, float], pd.Series]:
        df = self._data
        storage_start_index = df.index[0]
        storage_end_index = df.index[-1]
        if isinstance(i, int):
            if storage_start_index > i or i > storage_end_index:
                raise IndexError(f"{i}: start index is {storage_start_index}")
            data = self.data[i]
            return i, data
        elif isinstance(i, slice):
            start_index = storage_start_index if i.start is None else i.start
            end_index = storage_end_index if i.stop is None else i.stop
            si = max(start_index, storage_start_index)
            if si > end_index or self.field not in df.columns:
                return pd.Series(dtype=np.float32)  # type: ignore
            data = df[self.field].tolist()
            result = data[si - storage_start_index : end_index - storage_start_index]
            return pd.Series(result, index=pd.RangeIndex(si, si + len(result)))  # type: ignore
        else:
            raise TypeError(f"type(i) = {type(i)}")

    def __len__(self) -> int:
        return len(self.data)


class TestMockData(unittest.TestCase):
    _setup_kwargs = {
        "calendar_provider": {
            "class": "LocalCalendarProvider",
            "module_path": "qlib.data.data",
            "kwargs": {"backend": {"class": "MockCalendarStorage", "module_path": "qlib.tests"}},
        },
        "instrument_provider": {
            "class": "LocalInstrumentProvider",
            "module_path": "qlib.data.data",
            "kwargs": {"backend": {"class": "MockInstrumentStorage", "module_path": "qlib.tests"}},
        },
        "feature_provider": {
            "class": "LocalFeatureProvider",
            "module_path": "qlib.data.data",
            "kwargs": {"backend": {"class": "MockFeatureStorage", "module_path": "qlib.tests"}},
        },
    }

    @classmethod
    def setUpClass(cls) -> None:

        provider_uri = "Not necessary."
        init(region=REG_TW, provider_uri=provider_uri, expression_cache=None, dataset_cache=None, **cls._setup_kwargs)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import tempfile
import unittest
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import pandas as pd

import qlib
from qlib.tests import TestAutoData

from qlib.data.storage.file_storage import (
    FileCalendarStorage as CalendarStorage,
    FileInstrumentStorage as InstrumentStorage,
    FileFeatureStorage as FeatureStorage,
)

_file_name = Path(__file__).name.split(".")[0]
DATA_DIR = Path(__file__).parent.joinpath(f"{_file_name}_data")
QLIB_DIR = DATA_DIR.joinpath("qlib")
QLIB_DIR.mkdir(exist_ok=True, parents=True)


class TestStorage(TestAutoData):
    def test_calendar_storage(self):
        calendar = CalendarStorage(freq="day", future=False, provider_uri=self.provider_uri)
        assert isinstance(calendar[:], Iterable), f"{calendar.__class__.__name__}.__getitem__(s: slice) is not Iterable"
        assert isinstance(calendar.data, Iterable), f"{calendar.__class__.__name__}.data is not Iterable"

        print(f"calendar[1: 5]: {calendar[1:5]}")
        print(f"calendar[0]: {calendar[0]}")
        print(f"calendar[-1]: {calendar[-1]}")

        calendar = CalendarStorage(freq="1min", future=False, provider_uri="not_found")
        with self.assertRaises(ValueError):
            print(calendar.data)

        with self.assertRaises(ValueError):
            print(calendar[:])

        with self.assertRaises(ValueError):
            print(calendar[0])

    def test_instrument_storage(self):
        """
        The meaning of instrument, such as CSI500:

            CSI500 composition changes:

                date            add         remove
                2005-01-01      SH600000
                2005-01-01      SH600001
                2005-01-01      SH600002
                2005-02-01      SH600003    SH600000
                2005-02-15      SH600000    SH600002

            Calendar:
                pd.date_range(start="2020-01-01", stop="2020-03-01", freq="1D")

            Instrument:
                symbol      start_time      end_time
                SH600000    2005-01-01      2005-01-31 (2005-02-01 Last trading day)
                SH600000    2005-02-15      2005-03-01
                SH600001    2005-01-01      2005-03-01
                SH600002    2005-01-01      2005-02-14 (2005-02-15 Last trading day)
                SH600003    2005-02-01      2005-03-01

            InstrumentStorage:
                {
                    "SH600000": [(2005-01-01, 2005-01-31), (2005-02-15, 2005-03-01)],
                    "SH600001": [(2005-01-01, 2005-03-01)],
                    "SH600002": [(2005-01-01, 2005-02-14)],
                    "SH600003": [(2005-02-01, 2005-03-01)],
                }

        """

        instrument = InstrumentStorage(market="csi300", provider_uri=self.provider_uri, freq="day")

        for inst, spans in instrument.data.items():
            assert isinstance(inst, str) and isinstance(
                spans, Iterable
            ), f"{instrument.__class__.__name__} value is not Iterable"
            for s_e in spans:
                assert (
                    isinstance(s_e, tuple) and len(s_e) == 2
                ), f"{instrument.__class__.__name__}.__getitem__(k) TypeError"

        print(f"instrument['SH600000']: {instrument['SH600000']}")

        instrument = InstrumentStorage(market="csi300", provider_uri="not_found", freq="day")
        with self.assertRaises(ValueError):
            print(instrument.data)

        with self.assertRaises(ValueError):
            print(instrument["sSH600000"])

    def test_feature_storage(self):
        """
        Calendar:
            pd.date_range(start="2005-01-01", stop="2005-03-01", freq="1D")

        Instrument:
            {
                "SH600000": [(2005-01-01, 2005-01-31), (2005-02-15, 2005-03-01)],
                "SH600001": [(2005-01-01, 2005-03-01)],
                "SH600002": [(2005-01-01, 2005-02-14)],
                "SH600003": [(2005-02-01, 2005-03-01)],
            }

        Feature:
            Stock data(close):
                            2005-01-01  ...   2005-02-01   ...   2005-02-14  2005-02-15  ...  2005-03-01
                SH600000     1          ...      3         ...      4           5               6
                SH600001     1          ...      4         ...      5           6               7
                SH600002     1          ...      5         ...      6           nan             nan
                SH600003     nan        ...      1         ...      2           3               4

            FeatureStorage(SH600000, close):

                [
                    (calendar.index("2005-01-01"), 1),
                    ...,
                    (calendar.index("2005-03-01"), 6)
                ]

                ====> [(0, 1), ..., (59, 6)]


            FeatureStorage(SH600002, close):

                [
                    (calendar.index("2005-01-01"), 1),
                    ...,
                    (calendar.index("2005-02-14"), 6)
                ]

                ===> [(0, 1), ..., (44, 6)]

            FeatureStorage(SH600003, close):

                [
                    (calendar.index("2005-02-01"), 1),
                    ...,
                    (calendar.index("2005-03-01"), 4)
                ]

                ===> [(31, 1), ..., (59, 4)]

        """

        feature = FeatureStorage(instrument="SZ300677", field="close", freq="day", provider_uri=self.provider_uri)

        with self.assertRaises(IndexError):
            print(feature[0])
        assert isinstance(
            feature[3049][1], (float, np.float32)
        ), f"{feature.__class__.__name__}.__getitem__(i: int) error"
        assert len(feature[3049:3052]) == 3, f"{feature.__class__.__name__}.__getitem__(s: slice) error"
        print(f"feature[3049: 3052]: \n{feature[3049: 3052]}")

        print(f"feature[:].tail(): \n{feature[:].tail()}")

        feature = FeatureStorage(instrument="SH600004", field="close", freq="day", provider_uri="not_fount")

        with self.assertRaises(ValueError):
            print(feature[0])
        with self.assertRaises(ValueError):
            print(feature[:].empty)
        with self.assertRaises(ValueError):
            print(feature.data.empty)


class TestInstrumentStorageRoundTrip(unittest.TestCase):
    """Self-contained InstrumentStorage read/write checks (uses a temporary provider, no market data)."""

    @classmethod
    def setUpClass(cls) -> None:
        # qlib.init is only needed to populate the global config defaults (e.g. `C.mount_path`)
        # consulted by the storage path manager; the storages below use their own provider_uri.
        cls._init_tmp = tempfile.TemporaryDirectory()
        init_path = Path(cls._init_tmp.name)
        init_path.joinpath("calendars").mkdir()
        init_path.joinpath("calendars", "day.txt").write_text("2020-01-02\n")
        init_path.joinpath("instruments").mkdir()
        init_path.joinpath("features").mkdir()
        qlib.init(provider_uri=cls._init_tmp.name, expression_cache=None, dataset_cache=None)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._init_tmp.cleanup()

    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        provider_path = Path(self._tmp_dir.name)
        # the supported frequencies of a provider are derived from the calendar file names
        provider_path.joinpath("calendars").mkdir()
        provider_path.joinpath("calendars", "day.txt").write_text("2020-01-02\n2020-01-03\n2020-01-06\n")
        provider_path.joinpath("instruments").mkdir()
        self.provider_path = provider_path

    def tearDown(self) -> None:
        self._tmp_dir.cleanup()

    def _write_market_file(self, market: str, content: str) -> None:
        self.provider_path.joinpath("instruments", f"{market}.txt").write_text(content)

    def test_symbols_named_like_na_sentinels(self):
        """
        Tickers literally named "NA" or "NULL" must be loaded as strings.

        Reading them as NaN merges distinct tickers into one bogus float key (see issue #1720,
        whose fix in #1736 covered exists_qlib_data but not this storage reader).
        """
        self._write_market_file(
            "na_market",
            "NA\t2020-01-02\t2020-01-06\nNULL\t2020-01-02\t2020-01-06\nSH600000\t2020-01-02\t2020-01-06\n",
        )
        instrument = InstrumentStorage(market="na_market", freq="day", provider_uri=self._tmp_dir.name)
        data = instrument.data

        assert set(data.keys()) == {"NA", "NULL", "SH600000"}, f"symbols were NA-parsed: {list(data.keys())}"
        for symbol, spans in data.items():
            assert isinstance(symbol, str), f"symbol {symbol!r} is not str"
            assert spans == [
                (pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-06"))
            ], f"wrong spans for {symbol!r}: {spans}"

    def test_write_read_round_trip(self):
        """
        Instruments written through the public mutation API must read back unchanged
        (`_read_instrument` expects the on-disk column order [symbol, start, end]).
        """
        self._write_market_file("rt_market", "SH600000\t2020-01-02\t2020-01-06\n")
        instrument = InstrumentStorage(market="rt_market", freq="day", provider_uri=self._tmp_dir.name)
        instrument["SH600001"] = [(pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-06"))]

        reread = InstrumentStorage(market="rt_market", freq="day", provider_uri=self._tmp_dir.name).data
        assert reread == {
            "SH600000": [(pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-06"))],
            "SH600001": [(pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-06"))],
        }, f"round-trip corrupted the instrument file: {reread}"

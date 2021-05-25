# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from pathlib import Path
from collections.abc import Iterable

import pytest
import numpy as np
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
        with pytest.raises(ValueError):
            print(calendar.data)

        with pytest.raises(ValueError):
            print(calendar[:])

        with pytest.raises(ValueError):
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

        instrument = InstrumentStorage(market="csi300", provider_uri=self.provider_uri)

        for inst, spans in instrument.data.items():
            assert isinstance(inst, str) and isinstance(
                spans, Iterable
            ), f"{instrument.__class__.__name__} value is not Iterable"
            for s_e in spans:
                assert (
                    isinstance(s_e, tuple) and len(s_e) == 2
                ), f"{instrument.__class__.__name__}.__getitem__(k) TypeError"

        print(f"instrument['SH600000']: {instrument['SH600000']}")

        instrument = InstrumentStorage(market="csi300", provider_uri="not_found")
        with pytest.raises(ValueError):
            print(instrument.data)

        with pytest.raises(ValueError):
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

        feature = FeatureStorage(instrument="SH600004", field="close", freq="day", provider_uri=self.provider_uri)

        with pytest.raises(IndexError):
            print(feature[0])
        assert isinstance(
            feature[815][1], (float, np.float32)
        ), f"{feature.__class__.__name__}.__getitem__(i: int) error"
        assert len(feature[815:818]) == 3, f"{feature.__class__.__name__}.__getitem__(s: slice) error"
        print(f"feature[815: 818]: \n{feature[815: 818]}")

        print(f"feature[:].tail(): \n{feature[:].tail()}")

        feature = FeatureStorage(instrument="SH600004", field="close", freq="day", provider_uri="not_fount")

        assert feature[0] == (None, None), "FeatureStorage does not exist, feature[i] should return `(None, None)`"
        assert feature[:].empty, "FeatureStorage does not exist, feature[:] should return `pd.Series(dtype=np.float32)`"
        assert (
            feature.data.empty
        ), "FeatureStorage does not exist, feature.data should return `pd.Series(dtype=np.float32)`"

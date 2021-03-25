# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

import pandas as pd


# TODO: set STORAGE_NAME
STORAGE_NAME = ""
STORAGE_FILE_PATH = Path("")
# TODO: set value
CALENDAR_URI = ""
INSTRUMENT_URI = ""
FEATURE_URI = ""


def get_module(module_path: Path):
    module_spec = spec_from_file_location("", module_path)
    module = module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


STORAGE_MODULE = get_module(STORAGE_FILE_PATH)


CalendarStorage = getattr(STORAGE_MODULE, f"{STORAGE_NAME.title()}CalendarStorage")
InstrumentStorage = getattr(STORAGE_MODULE, f"{STORAGE_NAME.title()}InstrumentStorage")
FeatureStorage = getattr(STORAGE_MODULE, f"{STORAGE_NAME.title()}FeatureStorage")


class TestCalendarStorage:
    def test_calendar_storage(self):
        # calendar value: pd.date_range(start="2005-01-01", stop="2005-03-01", freq="1D")
        start_date = "2005-01-01"
        end_date = "2005-03-01"
        values = pd.date_range(start_date, end_date, freq="1D")

        calendar = CalendarStorage(uri=CALENDAR_URI)
        # test `__iter__`
        for _s, _t in zip(calendar, values):
            assert pd.Timestamp(_s) == pd.Timestamp(_t), f"{calendar.__name__}.__iter__ error"

        # test `__getitem__(self, s: slice)`
        for _s, _t in zip(calendar[1:3], values[1:3]):
            assert pd.Timestamp(_s) == pd.Timestamp(_t), f"{calendar.__name__}.__getitem__(s: slice) error"

        # test `__getitem__(self, i)`
        assert pd.Timestamp(calendar[0]) == pd.Timestamp(values[0]), f"{calendar.__name__}.__getitem__(i: int) error"

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
        base_instrument = {
            "SH600000": [("2005-01-01", "2005-01-31"), ("2005-02-15", "2005-03-01")],
            "SH600001": [("2005-01-01", "2005-03-01")],
            "SH600002": [("2005-01-01", "2005-02-14")],
            "SH600003": [("2005-02-01", "2005-03-01")],
        }
        instrument = InstrumentStorage(uri=INSTRUMENT_URI)

        # test `keys`
        assert sorted(instrument.keys()) == sorted(base_instrument.keys()), f"{instrument.__name__}.keys error"
        # test `__getitem__`
        assert instrument["SH600000"] == base_instrument["SH600000"], f"{instrument.__name__}.__getitem__ error"
        # test `get`
        assert instrument.get("SH600001") == base_instrument.get("SH600001"), f"{instrument.__name__}.get error"
        # test `items`
        for _item in instrument.items():
            assert base_instrument[_item[0]] == _item[1]
        assert len(instrument.items()) == len(instrument) == len(base_instrument), f"{instrument.__name__}.items error"

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

        # FeatureStorage(SH600003, close)
        feature = FeatureStorage(uri=FEATURE_URI)
        # 2005-02-01 and 2005-03-01
        assert feature[31] == 1 and feature[59] == 4, f"{feature.__name__}.__getitem__(i: int) error"

        # 2005-02-01, 2005-02-02, 2005-02-03
        # close_items: [(31, 1), ..., (33, <value>)]
        close_items = feature[31:34]

        # 2005-02-01, ..., 2005-03-01
        # feature: [(31, 1), ..., (59, 4)]
        print(feature)

        assert (
            len(feature) == len(feature[:]) == len(feature[31:60]) == 29
        ), f"{feature.__name__}.items/__getitem__(s: slice) error"

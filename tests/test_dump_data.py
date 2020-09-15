#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.


import sys
import shutil
import unittest
from pathlib import Path

import qlib
import numpy as np
import pandas as pd
from qlib.data import D

sys.path.append(str(Path(__file__).resolve().parent.parent.joinpath("scripts")))
from get_data import GetData
from dump_bin import DumpData


DATA_DIR = Path(__file__).parent.joinpath("test_data")
DATA_DIR.mkdir(exist_ok=True, parents=True)
SOURCE_DIR = DATA_DIR.joinpath("source")
QLIB_DIR = DATA_DIR.joinpath("qlib")
QLIB_DIR.mkdir(exist_ok=True, parents=True)


class TestDumpData(unittest.TestCase):
    FIELDS = "open,close,high,low,volume,vwap,factor,change,money".split(",")
    QLIB_FIELDS = list(map(lambda x: f"${x}", FIELDS))
    DUMP_DATA = DumpData(csv_path=SOURCE_DIR, qlib_dir=QLIB_DIR)
    SOTCK_NAMES = list(map(lambda x: x.name[:-4].upper(), SOURCE_DIR.iterdir()))

    # simpe data
    SIMPLE_DATA = None

    @classmethod
    def setUpClass(cls) -> None:
        GetData().csv_data_cn(SOURCE_DIR)
        mount_path = provider_uri = str(QLIB_DIR.resolve())
        qlib.init(
            mount_path=mount_path,
            provider_uri=provider_uri,
            expression_cache=None,
            dataset_cache=None,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(str(DATA_DIR.resolve()))

    def test_0_dump_calendars(self):
        self.DUMP_DATA.dump_calendars()
        ori_calendars = set(
            map(
                pd.Timestamp,
                pd.read_csv(QLIB_DIR.joinpath("calendars", "day.txt"), header=None).loc[:, 0].values,
            )
        )
        res_calendars = set(D.calendar())
        assert len(ori_calendars - res_calendars) == len(res_calendars - ori_calendars) == 0, "dump calendars failed"

    def test_1_dump_instruments(self):
        self.DUMP_DATA.dump_instruments()
        ori_ins = set(map(lambda x: x.name[:-4].upper(), SOURCE_DIR.iterdir()))
        res_ins = set(D.list_instruments(D.instruments("all"), as_list=True))
        assert len(ori_ins - res_ins) == len(ori_ins - res_ins) == 0, "dump instruments failed"

    def test_2_dump_features(self):
        self.DUMP_DATA.dump_features(include_fields=self.FIELDS)
        df = D.features(self.SOTCK_NAMES, self.QLIB_FIELDS)
        TestDumpData.SIMPLE_DATA = df.loc(axis=0)[self.SOTCK_NAMES[0], :]
        self.assertFalse(df.dropna().empty, "features data failed")
        self.assertListEqual(list(df.columns), self.QLIB_FIELDS, "features columns failed")

    def test_3_dump_features_simple(self):
        stock = self.SOTCK_NAMES[0]
        dump_data = DumpData(csv_path=SOURCE_DIR.joinpath(f"{stock.upper()}.csv"), qlib_dir=QLIB_DIR)
        dump_data.dump_features(include_fields=self.FIELDS, calendar_path=QLIB_DIR.joinpath("calendars", "day.txt"))

        df = D.features([stock], self.QLIB_FIELDS)

        self.assertEqual(len(df), len(TestDumpData.SIMPLE_DATA), "dump features simple failed")
        self.assertTrue(np.isclose(df.dropna(), self.SIMPLE_DATA.dropna()).all(), "dump features simple failed")


if __name__ == "__main__":
    unittest.main()

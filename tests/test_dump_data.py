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
from dump_bin import DumpDataAll, DumpDataFix


DATA_DIR = Path(__file__).parent.joinpath("test_dump_data")
SOURCE_DIR = DATA_DIR.joinpath("source")
SOURCE_DIR.mkdir(exist_ok=True, parents=True)
QLIB_DIR = DATA_DIR.joinpath("qlib")
QLIB_DIR.mkdir(exist_ok=True, parents=True)


class TestDumpData(unittest.TestCase):
    FIELDS = "open,close,high,low,volume".split(",")
    QLIB_FIELDS = list(map(lambda x: f"${x}", FIELDS))
    DUMP_DATA = None
    STOCK_NAMES = None

    # simpe data
    SIMPLE_DATA = None

    @classmethod
    def setUpClass(cls) -> None:
        GetData().download_data(file_name="csv_data_cn.zip", target_dir=SOURCE_DIR)
        TestDumpData.DUMP_DATA = DumpDataAll(csv_path=SOURCE_DIR, qlib_dir=QLIB_DIR, include_fields=cls.FIELDS)
        TestDumpData.STOCK_NAMES = list(map(lambda x: x.name[:-4].upper(), SOURCE_DIR.glob("*.csv")))
        provider_uri = str(QLIB_DIR.resolve())
        qlib.init(
            provider_uri=provider_uri,
            expression_cache=None,
            dataset_cache=None,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(str(DATA_DIR.resolve()))

    def test_0_dump_bin(self):
        self.DUMP_DATA.dump()

    def test_1_dump_calendars(self):
        ori_calendars = set(
            map(
                pd.Timestamp,
                pd.read_csv(QLIB_DIR.joinpath("calendars", "day.txt"), header=None).loc[:, 0].values,
            )
        )
        res_calendars = set(D.calendar())
        assert len(ori_calendars - res_calendars) == len(res_calendars - ori_calendars) == 0, "dump calendars failed"

    def test_2_dump_instruments(self):
        ori_ins = set(map(lambda x: x.name[:-4].upper(), SOURCE_DIR.glob("*.csv")))
        res_ins = set(D.list_instruments(D.instruments("all"), as_list=True))
        assert len(ori_ins - res_ins) == len(ori_ins - res_ins) == 0, "dump instruments failed"

    def test_3_dump_features(self):
        df = D.features(self.STOCK_NAMES, self.QLIB_FIELDS)
        TestDumpData.SIMPLE_DATA = df.loc(axis=0)[self.STOCK_NAMES[0], :]
        self.assertFalse(df.dropna().empty, "features data failed")
        self.assertListEqual(list(df.columns), self.QLIB_FIELDS, "features columns failed")

    def test_4_dump_features_simple(self):
        stock = self.STOCK_NAMES[0]
        dump_data = DumpDataFix(
            csv_path=SOURCE_DIR.joinpath(f"{stock.lower()}.csv"), qlib_dir=QLIB_DIR, include_fields=self.FIELDS
        )
        dump_data.dump()

        df = D.features([stock], self.QLIB_FIELDS)

        self.assertEqual(len(df), len(TestDumpData.SIMPLE_DATA), "dump features simple failed")
        self.assertTrue(np.isclose(df.dropna(), self.SIMPLE_DATA.dropna()).all(), "dump features simple failed")


if __name__ == "__main__":
    unittest.main()

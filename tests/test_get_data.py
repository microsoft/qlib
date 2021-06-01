#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import shutil
import unittest
from pathlib import Path

import qlib
from qlib.data import D
from qlib.tests.data import GetData

DATA_DIR = Path(__file__).parent.joinpath("test_get_data")
SOURCE_DIR = DATA_DIR.joinpath("source")
SOURCE_DIR.mkdir(exist_ok=True, parents=True)
QLIB_DIR = DATA_DIR.joinpath("qlib")
QLIB_DIR.mkdir(exist_ok=True, parents=True)


class TestGetData(unittest.TestCase):
    FIELDS = "$open,$close,$high,$low,$volume,$factor,$change".split(",")

    @classmethod
    def setUpClass(cls) -> None:
        provider_uri = str(QLIB_DIR.resolve())
        qlib.init(
            provider_uri=provider_uri,
            expression_cache=None,
            dataset_cache=None,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(str(DATA_DIR.resolve()))

    def test_0_qlib_data(self):

        GetData().qlib_data(
            name="qlib_data_simple", target_dir=QLIB_DIR, region="cn", interval="1d", delete_old=False, exists_skip=True
        )
        df = D.features(D.instruments("csi300"), self.FIELDS)
        self.assertListEqual(list(df.columns), self.FIELDS, "get qlib data failed")
        self.assertFalse(df.dropna().empty, "get qlib data failed")

    def test_1_csv_data(self):
        GetData().csv_data_cn(SOURCE_DIR)
        stock_name = set(map(lambda x: x.name[:-4].upper(), SOURCE_DIR.glob("*.csv")))
        self.assertEqual(len(stock_name), 85, "get csv data failed")


if __name__ == "__main__":
    unittest.main()

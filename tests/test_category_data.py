import unittest
import pathlib
import shutil

import pandas as pd

import qlib
from qlib.data import D
from qlib.tests.utils import split_df_to_str
from scripts.dump_bin import DumpDataAll, DumpDataFix, DumpDataUpdate


pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", None)

TEST_DATA_DIR = pathlib.Path(__file__).parent / "test_dump_category_data"
BASE_SOURCE_DATA_DIR = TEST_DATA_DIR / "source"
SOURCE_DATA_DIR_ALL = BASE_SOURCE_DATA_DIR / "all"
SOURCE_DATA_DIR_FIX = BASE_SOURCE_DATA_DIR / "fix"
SOURCE_DATA_DIR_UPDATE = BASE_SOURCE_DATA_DIR / "update"
QLIB_DIR = TEST_DATA_DIR / "qlib"


class TestCategoryData(unittest.TestCase):
    SAMPLE_DATA_ALL = [
        {
            "symbol": "sh600519",
            "date": "2022-08-01",
            "open": 1890.01,
            "close": 1890.3,
            "name": "贵州茅台",
            "industry": "食品饮料",
        },
        {
            "symbol": "sh600519",
            "date": "2022-08-02",
            "open": 1880,
            "close": 1879.98,
            "name": "贵州茅台",
            "industry": "食品饮料",
        },
        {
            "symbol": "sh600519",
            "date": "2022-08-03",
            "open": 1889.99,
            "close": 1885,
            "name": "贵州茅台",
            "industry": "食品饮料",
        },
        {
            "symbol": "sh600519",
            "date": "2022-08-04",
            "open": 1890,
            "close": 1916.01,
            "name": "贵州茅台",
            "industry": "食品饮料",
        },
        {
            "symbol": "sh600519",
            "date": "2022-08-05",
            "open": 1927.8,
            "close": 1923.96,
            "name": "贵州茅台",
            "industry": "食品饮料",
        },
    ]

    SAMPLE_DATA_FIX = [
        {"symbol": "sh601127", "date": "2022-08-01", "open": 75.9, "close": 81.03, "name": "小康股份", "industry": "汽车"},
        {"symbol": "sh601127", "date": "2022-08-02", "open": 78, "close": 74.56, "name": "赛力斯", "industry": "汽车"},
        {"symbol": "sh601127", "date": "2022-08-03", "open": 75, "close": 70.91, "name": "赛力斯", "industry": "汽车"},
        {"symbol": "sh601127", "date": "2022-08-04", "open": 71.08, "close": 68.1, "name": "赛力斯", "industry": "汽车"},
    ]

    SAMPLE_DATA_UPDATE = [
        {"symbol": "sh601127", "date": "2022-08-05", "open": 68.11, "close": 67.38, "name": "赛力斯", "industry": "汽车"},
    ]
    INSTRUMENTS = ["sh600519", "sh601127"]
    FIELD = ["$open", "$close", "$name", "$industry"]

    @staticmethod
    def reinit_qlib():
        provider_uri = str(QLIB_DIR.resolve())
        qlib.init(
            provider_uri=provider_uri,
            expression_cache=None,
            dataset_cache=None,
        )

    @classmethod
    def setUpClass(cls) -> None:
        QLIB_DIR.mkdir(exist_ok=True, parents=True)

        SOURCE_DATA_DIR_ALL.mkdir(exist_ok=True, parents=True)
        SOURCE_DATA_DIR_FIX.mkdir(exist_ok=True, parents=True)
        SOURCE_DATA_DIR_UPDATE.mkdir(exist_ok=True, parents=True)
        for symbol, group_df in pd.DataFrame(cls.SAMPLE_DATA_ALL).groupby("symbol"):
            group_df.to_csv(SOURCE_DATA_DIR_ALL / f"{symbol}.csv", index=False)

        for symbol, group_df in pd.DataFrame(cls.SAMPLE_DATA_FIX).groupby("symbol"):
            group_df.to_csv(SOURCE_DATA_DIR_FIX / f"{symbol}.csv", index=False)

        for symbol, group_df in pd.DataFrame(cls.SAMPLE_DATA_UPDATE).groupby("symbol"):
            group_df.to_csv(SOURCE_DATA_DIR_UPDATE / f"{symbol}.csv", index=False)
        cls.reinit_qlib()
        DumpDataAll(
            csv_path=str(SOURCE_DATA_DIR_ALL.resolve()),
            qlib_dir=str(QLIB_DIR.resolve()),
            exclude_fields="symbol,date",
        )()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(str(TEST_DATA_DIR.resolve()))

    def test_dump_data(self) -> None:
        dump_all_df = D.features(self.INSTRUMENTS, self.FIELD)
        except_dump_all_df = """
                                     $open       $close  $name  $industry
        instrument datetime
        sh600519   2022-08-01  1890.010010  1890.300049       贵州茅台           食品饮料
                   2022-08-02  1880.000000  1879.979980       贵州茅台           食品饮料
                   2022-08-03  1889.989990  1885.000000       贵州茅台           食品饮料
                   2022-08-04  1890.000000  1916.010010       贵州茅台           食品饮料
                   2022-08-05  1927.800049  1923.959961       贵州茅台           食品饮料
        """
        self.assertEqual(split_df_to_str(dump_all_df), split_df_to_str(except_dump_all_df))
        DumpDataFix(
            csv_path=str(SOURCE_DATA_DIR_FIX.resolve()),
            qlib_dir=str(QLIB_DIR.resolve()),
            exclude_fields="symbol,date",
        )()
        self.reinit_qlib()
        dump_fix_df = D.features(self.INSTRUMENTS, self.FIELD)
        except_dump_fix_df = """
                                     $open       $close  $name  $industry
        instrument datetime
        sh600519   2022-08-01  1890.010010  1890.300049       贵州茅台           食品饮料
                   2022-08-02  1880.000000  1879.979980       贵州茅台           食品饮料
                   2022-08-03  1889.989990  1885.000000       贵州茅台           食品饮料
                   2022-08-04  1890.000000  1916.010010       贵州茅台           食品饮料
                   2022-08-05  1927.800049  1923.959961       贵州茅台           食品饮料
        sh601127   2022-08-01    75.900002    81.029999       小康股份             汽车
                   2022-08-02    78.000000    74.559998        赛力斯             汽车
                   2022-08-03    75.000000    70.910004        赛力斯             汽车
                   2022-08-04    71.080002    68.099998        赛力斯             汽车
        """
        self.assertEqual(split_df_to_str(dump_fix_df), split_df_to_str(except_dump_fix_df))
        DumpDataUpdate(
            csv_path=str(SOURCE_DATA_DIR_UPDATE.resolve()),
            qlib_dir=str(QLIB_DIR.resolve()),
            exclude_fields="symbol,date",
        )()
        self.reinit_qlib()
        dump_update_df = D.features(self.INSTRUMENTS, self.FIELD)
        except_dump_update_df = """
                                     $open       $close  $name  $industry
        instrument datetime
        sh600519   2022-08-01  1890.010010  1890.300049       贵州茅台           食品饮料
                   2022-08-02  1880.000000  1879.979980       贵州茅台           食品饮料
                   2022-08-03  1889.989990  1885.000000       贵州茅台           食品饮料
                   2022-08-04  1890.000000  1916.010010       贵州茅台           食品饮料
                   2022-08-05  1927.800049  1923.959961       贵州茅台           食品饮料
        sh601127   2022-08-01    75.900002    81.029999       小康股份             汽车
                   2022-08-02    78.000000    74.559998        赛力斯             汽车
                   2022-08-03    75.000000    70.910004        赛力斯             汽车
                   2022-08-04    71.080002    68.099998        赛力斯             汽车
                   2022-08-05    68.110001    67.379997        赛力斯             汽车
        """
        self.assertEqual(split_df_to_str(dump_update_df), split_df_to_str(except_dump_update_df))

    def test_base_opt(self):
        fields = self.FIELD + ["Ref($name, 1)"]
        df = D.features(["sh600519"], fields)
        except_df = """
                                     $open       $close $name $industry Ref($name, 1)
        instrument datetime
        sh600519   2022-08-01  1890.010010  1890.300049  贵州茅台      食品饮料           NaN
                   2022-08-02  1880.000000  1879.979980  贵州茅台      食品饮料          贵州茅台
                   2022-08-03  1889.989990  1885.000000  贵州茅台      食品饮料          贵州茅台
                   2022-08-04  1890.000000  1916.010010  贵州茅台      食品饮料          贵州茅台
                   2022-08-05  1927.800049  1923.959961  贵州茅台      食品饮料          贵州茅台
        """
        self.assertEqual(split_df_to_str(df), split_df_to_str(except_df))

    def test_np_opt(self):
        fields = self.FIELD + ["Abs($name)"]
        with self.assertRaises(ValueError) as context:
            D.features(["sh600519"], fields)
        self.assertEqual("Numpy element-wise operator only support float dtype.", context.exception.args[0])

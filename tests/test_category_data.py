import unittest
import pathlib
import shutil

import pandas as pd

import qlib
from qlib.data import D
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

        provider_uri = str(QLIB_DIR.resolve())
        qlib.init(
            provider_uri=provider_uri,
            expression_cache=None,
            dataset_cache=None,
        )

        DumpDataAll(
            csv_path=str(SOURCE_DATA_DIR_ALL.resolve()),
            qlib_dir=str(QLIB_DIR.resolve()),
            exclude_fields="symbol,date",
            is_convert_category=True,
        )()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(str(TEST_DATA_DIR.resolve()))

    @staticmethod
    def to_str(df):
        return "".join(str(df).split())

    def assert_eq(self, a, b):
        self.assertEqual(self.to_str(a), self.to_str(b))

    def test_dump_data(self) -> None:
        fields = self.FIELD + ["Cat($name)", "Cat($industry)"]
        dump_all_df = D.features(self.INSTRUMENTS, fields)
        except_dump_all_df = """
                                     $open       $close  $name  $industry Cat($name) Cat($industry)
        instrument datetime
        sh600519   2022-08-01  1890.010010  1890.300049    0.0        0.0       贵州茅台           食品饮料
                   2022-08-02  1880.000000  1879.979980    0.0        0.0       贵州茅台           食品饮料
                   2022-08-03  1889.989990  1885.000000    0.0        0.0       贵州茅台           食品饮料
                   2022-08-04  1890.000000  1916.010010    0.0        0.0       贵州茅台           食品饮料
                   2022-08-05  1927.800049  1923.959961    0.0        0.0       贵州茅台           食品饮料
        """
        self.assert_eq(dump_all_df, except_dump_all_df)
        DumpDataFix(
            csv_path=str(SOURCE_DATA_DIR_FIX.resolve()),
            qlib_dir=str(QLIB_DIR.resolve()),
            exclude_fields="symbol,date",
            is_convert_category=True,
        )()
        dump_fix_df = D.features(self.INSTRUMENTS, fields)
        except_dump_fix_df = """
                                     $open       $close  $name  $industry Cat($name) Cat($industry)
        instrument datetime
        sh600519   2022-08-01  1890.010010  1890.300049    0.0        0.0       贵州茅台           食品饮料
                   2022-08-02  1880.000000  1879.979980    0.0        0.0       贵州茅台           食品饮料
                   2022-08-03  1889.989990  1885.000000    0.0        0.0       贵州茅台           食品饮料
                   2022-08-04  1890.000000  1916.010010    0.0        0.0       贵州茅台           食品饮料
                   2022-08-05  1927.800049  1923.959961    0.0        0.0       贵州茅台           食品饮料
        sh601127   2022-08-01    75.900002    81.029999    1.0        1.0       小康股份             汽车
                   2022-08-02    78.000000    74.559998    2.0        1.0        赛力斯             汽车
                   2022-08-03    75.000000    70.910004    2.0        1.0        赛力斯             汽车
                   2022-08-04    71.080002    68.099998    2.0        1.0        赛力斯             汽车
        """
        self.assert_eq(dump_fix_df, except_dump_fix_df)
        DumpDataUpdate(
            csv_path=str(SOURCE_DATA_DIR_UPDATE.resolve()),
            qlib_dir=str(QLIB_DIR.resolve()),
            exclude_fields="symbol,date",
            is_convert_category=True,
        )()
        dump_update_df = D.features(self.INSTRUMENTS, fields)
        except_dump_update_df = """
                                     $open       $close  $name  $industry Cat($name) Cat($industry)
        instrument datetime
        sh600519   2022-08-01  1890.010010  1890.300049    0.0        0.0       贵州茅台           食品饮料
                   2022-08-02  1880.000000  1879.979980    0.0        0.0       贵州茅台           食品饮料
                   2022-08-03  1889.989990  1885.000000    0.0        0.0       贵州茅台           食品饮料
                   2022-08-04  1890.000000  1916.010010    0.0        0.0       贵州茅台           食品饮料
                   2022-08-05  1927.800049  1923.959961    0.0        0.0       贵州茅台           食品饮料
        sh601127   2022-08-01    75.900002    81.029999    1.0        1.0       小康股份             汽车
                   2022-08-02    78.000000    74.559998    2.0        1.0        赛力斯             汽车
                   2022-08-03    75.000000    70.910004    2.0        1.0        赛力斯             汽车
                   2022-08-04    71.080002    68.099998    2.0        1.0        赛力斯             汽车
                   2022-08-05    68.110001    67.379997    2.0        1.0        赛力斯             汽车
        """
        self.assert_eq(dump_update_df, except_dump_update_df)

    def test_base_opt(self):
        fields = self.FIELD + ["Cat($name)", "Ref(Cat($name), 1)"]
        df = D.features(["sh600519"], fields)
        except_df = """
                                     $open       $close  $name  $industry Cat($name) Ref(Cat($name), 1)
        instrument datetime
        sh600519   2022-08-01  1890.010010  1890.300049    0.0        0.0       贵州茅台                NaN
                   2022-08-02  1880.000000  1879.979980    0.0        0.0       贵州茅台               贵州茅台
                   2022-08-03  1889.989990  1885.000000    0.0        0.0       贵州茅台               贵州茅台
                   2022-08-04  1890.000000  1916.010010    0.0        0.0       贵州茅台               贵州茅台
                   2022-08-05  1927.800049  1923.959961    0.0        0.0       贵州茅台               贵州茅台
        """
        self.assert_eq(df, except_df)

    def test_np_opt(self):
        fields = self.FIELD + ["Cat($name)", "Abs(Cat($name))"]
        with self.assertRaises(ValueError) as context:
            D.features(["sh600519"], fields)
        self.assertTrue("Numpy element-wise operator only support float32 dtype." in context.exception.args)

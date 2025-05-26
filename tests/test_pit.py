# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import sys
import qlib
import shutil
import unittest
import pytest
import pandas as pd
from pathlib import Path

from qlib.data import D
from qlib.tests.data import GetData

sys.path.append(str(Path(__file__).resolve().parent.parent.joinpath("scripts")))
from dump_pit import DumpPitData

sys.path.append(str(Path(__file__).resolve().parent.parent.joinpath("scripts/data_collector/pit")))
from collector import Run


pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", None)

DATA_DIR = Path(__file__).parent.joinpath("test_pit_data")
SOURCE_DIR = DATA_DIR.joinpath("stock_data/source")
SOURCE_DIR.mkdir(exist_ok=True, parents=True)
QLIB_DIR = DATA_DIR.joinpath("qlib_data")
QLIB_DIR.mkdir(exist_ok=True, parents=True)


class TestPIT(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(str(DATA_DIR.resolve()))

    @classmethod
    def setUpClass(cls) -> None:
        cn_data_dir = str(QLIB_DIR.joinpath("cn_data").resolve())
        pit_dir = str(SOURCE_DIR.joinpath("pit").resolve())
        pit_normalized_dir = str(SOURCE_DIR.joinpath("pit_normalized").resolve())
        GetData().qlib_data(
            name="qlib_data_simple", target_dir=cn_data_dir, region="cn", delete_old=False, exists_skip=True
        )
        GetData().qlib_data(name="qlib_data", target_dir=pit_dir, region="pit", delete_old=False, exists_skip=True)

        # NOTE: This code does the same thing as line 43, but since baostock is not stable in downloading data, we have chosen to download offline data.
        # bs.login()
        # Run(
        #     source_dir=pit_dir,
        #     interval="quarterly",
        # ).download_data(start="2000-01-01", end="2020-01-01", symbol_regex="^(600519|000725).*")
        # bs.logout()

        Run(
            source_dir=pit_dir,
            normalize_dir=pit_normalized_dir,
            interval="quarterly",
        ).normalize_data()
        DumpPitData(
            csv_path=pit_normalized_dir,
            qlib_dir=cn_data_dir,
        ).dump(interval="quarterly")

    def setUp(self):
        # qlib.init(kernels=1)  # NOTE: set kernel to 1 to make it debug easier
        provider_uri = str(QLIB_DIR.joinpath("cn_data").resolve())
        qlib.init(provider_uri=provider_uri)

    def to_str(self, obj):
        return "".join(str(obj).split())

    def check_same(self, a, b):
        self.assertEqual(self.to_str(a), self.to_str(b))

    def test_query(self):
        instruments = ["sh600519"]
        fields = ["P($$roewa_q)", "P($$yoyni_q)"]
        # Mao Tai published 2019Q2 report at 2019-07-13 & 2019-07-18
        # - http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search&lastPage=index
        data = D.features(instruments, fields, start_time="2019-01-01", end_time="2019-07-19", freq="day")
        res = """
               P($$roewa_q)  P($$yoyni_q)
        count    133.000000    133.000000
        mean       0.196412      0.277930
        std        0.097591      0.030262
        min        0.000000      0.243892
        25%        0.094737      0.243892
        50%        0.255220      0.304181
        75%        0.255220      0.305041
        max        0.344644      0.305041
        """
        self.check_same(data.describe(), res)

        res = """
                               P($$roewa_q)  P($$yoyni_q)
        instrument datetime
        sh600519   2019-07-15      0.000000      0.305041
                   2019-07-16      0.000000      0.305041
                   2019-07-17      0.000000      0.305041
                   2019-07-18      0.175322      0.252650
                   2019-07-19      0.175322      0.252650
        """
        self.check_same(data.tail(), res)

    def test_no_exist_data(self):
        fields = ["P($$roewa_q)", "P($$yoyni_q)", "$close"]
        data = D.features(["sh600519", "sh601988"], fields, start_time="2019-01-01", end_time="2019-07-19", freq="day")
        data["$close"] = 1  # in case of different dataset gives different values
        expect = """
                               P($$roewa_q)  P($$yoyni_q)  $close
        instrument datetime
        sh600519   2019-01-02       0.25522      0.243892       1
                   2019-01-03       0.25522      0.243892       1
                   2019-01-04       0.25522      0.243892       1
                   2019-01-07       0.25522      0.243892       1
                   2019-01-08       0.25522      0.243892       1
        ...                             ...           ...     ...
        sh601988   2019-07-15           NaN           NaN       1
                   2019-07-16           NaN           NaN       1
                   2019-07-17           NaN           NaN       1
                   2019-07-18           NaN           NaN       1
                   2019-07-19           NaN           NaN       1

        [266 rows x 3 columns]
        """
        self.check_same(data, expect)

    @pytest.mark.slow
    def test_expr(self):
        fields = [
            "P(Mean($$roewa_q, 1))",
            "P($$roewa_q)",
            "P(Mean($$roewa_q, 2))",
            "P(Ref($$roewa_q, 1))",
            "P((Ref($$roewa_q, 1) +$$roewa_q) / 2)",
        ]
        instruments = ["sh600519"]
        data = D.features(instruments, fields, start_time="2019-01-01", end_time="2019-07-19", freq="day")
        expect = """
                               P(Mean($$roewa_q, 1))  P($$roewa_q)  P(Mean($$roewa_q, 2))  P(Ref($$roewa_q, 1))  P((Ref($$roewa_q, 1) +$$roewa_q) / 2)
        instrument datetime
        sh600519   2019-07-01               0.094737      0.094737               0.219691              0.344644                               0.219691
                   2019-07-02               0.094737      0.094737               0.219691              0.344644                               0.219691
                   2019-07-03               0.094737      0.094737               0.219691              0.344644                               0.219691
                   2019-07-04               0.094737      0.094737               0.219691              0.344644                               0.219691
                   2019-07-05               0.094737      0.094737               0.219691              0.344644                               0.219691
                   2019-07-08               0.094737      0.094737               0.219691              0.344644                               0.219691
                   2019-07-09               0.094737      0.094737               0.219691              0.344644                               0.219691
                   2019-07-10               0.094737      0.094737               0.219691              0.344644                               0.219691
                   2019-07-11               0.094737      0.094737               0.219691              0.344644                               0.219691
                   2019-07-12               0.094737      0.094737               0.219691              0.344644                               0.219691
                   2019-07-15               0.000000      0.000000               0.047369              0.094737                               0.047369
                   2019-07-16               0.000000      0.000000               0.047369              0.094737                               0.047369
                   2019-07-17               0.000000      0.000000               0.047369              0.094737                               0.047369
                   2019-07-18               0.175322      0.175322               0.135029              0.094737                               0.135029
                   2019-07-19               0.175322      0.175322               0.135029              0.094737                               0.135029
        """
        self.check_same(data.tail(15), expect)

    def test_unlimit(self):
        # fields = ["P(Mean($$roewa_q, 1))", "P($$roewa_q)", "P(Mean($$roewa_q, 2))", "P(Ref($$roewa_q, 1))", "P((Ref($$roewa_q, 1) +$$roewa_q) / 2)"]
        fields = ["P($$roewa_q)"]
        instruments = ["sh600519"]
        _ = D.features(instruments, fields, freq="day")  # this should not raise error
        data = D.features(instruments, fields, end_time="2020-01-01", freq="day")  # this should not raise error
        s = data.iloc[:, 0]
        # You can check the expected value based on the content in `docs/advanced/PIT.rst`
        expect = """
        instrument  datetime
        sh600519    2005-01-04         NaN
                    2007-04-30    0.090219
                    2007-08-17    0.139330
                    2007-10-23    0.245863
                    2008-03-03    0.347900
                    2008-03-13    0.395989
                    2008-04-22    0.100724
                    2008-08-28    0.249968
                    2008-10-27    0.334120
                    2009-03-25    0.390117
                    2009-04-21    0.102675
                    2009-08-07    0.230712
                    2009-10-26    0.300730
                    2010-04-02    0.335461
                    2010-04-26    0.083825
                    2010-08-12    0.200545
                    2010-10-29    0.260986
                    2011-03-21    0.307393
                    2011-04-25    0.097411
                    2011-08-31    0.248251
                    2011-10-18    0.318919
                    2012-03-23    0.403900
                    2012-04-11    0.403925
                    2012-04-26    0.112148
                    2012-08-10    0.264847
                    2012-10-26    0.370487
                    2013-03-29    0.450047
                    2013-04-18    0.099958
                    2013-09-02    0.210442
                    2013-10-16    0.304543
                    2014-03-25    0.394328
                    2014-04-25    0.083217
                    2014-08-29    0.164503
                    2014-10-30    0.234085
                    2015-04-21    0.078494
                    2015-08-28    0.137504
                    2015-10-23    0.201709
                    2016-03-24    0.264205
                    2016-04-21    0.073664
                    2016-08-29    0.136576
                    2016-10-31    0.188062
                    2017-04-17    0.244385
                    2017-04-25    0.080614
                    2017-07-28    0.151510
                    2017-10-26    0.254166
                    2018-03-28    0.329542
                    2018-05-02    0.088887
                    2018-08-02    0.170563
                    2018-10-29    0.255220
                    2019-03-29    0.344644
                    2019-04-25    0.094737
                    2019-07-15    0.000000
                    2019-07-18    0.175322
                    2019-10-16    0.255819
        Name: P($$roewa_q), dtype: float32
        """
        self.check_same(s[~s.duplicated().values], expect)

    def test_expr2(self):
        instruments = ["sh600519"]
        fields = ["P($$roewa_q)", "P($$yoyni_q)"]
        fields += ["P(($$roewa_q / $$yoyni_q) / Ref($$roewa_q / $$yoyni_q, 1) - 1)"]
        fields += ["P(Sum($$yoyni_q, 4))"]
        fields += ["$close", "P($$roewa_q) * $close"]
        data = D.features(instruments, fields, start_time="2019-01-01", end_time="2020-01-01", freq="day")
        except_data = """
                                       P($$roewa_q)  P($$yoyni_q)  P(($$roewa_q / $$yoyni_q) / Ref($$roewa_q / $$yoyni_q, 1) - 1)  P(Sum($$yoyni_q, 4))      $close  P($$roewa_q) * $close
        instrument datetime
        sh600519   2019-01-02      0.255220      0.243892                                           1.484224                           1.661578   63.595333              16.230801
                   2019-01-03      0.255220      0.243892                                           1.484224                           1.661578   62.641907              15.987467
                   2019-01-04      0.255220      0.243892                                           1.484224                           1.661578   63.915985              16.312637
                   2019-01-07      0.255220      0.243892                                           1.484224                           1.661578   64.286530              16.407207
                   2019-01-08      0.255220      0.243892                                           1.484224                           1.661578   64.212196              16.388237
        ...                             ...           ...                                                ...                                ...         ...                    ...
                   2019-12-25      0.255819      0.219821                                           0.677052                           1.081693  122.150467              31.248409
                   2019-12-26      0.255819      0.219821                                           0.677052                           1.081693  122.301315              31.286999
                   2019-12-27      0.255819      0.219821                                           0.677052                           1.081693  125.307404              32.056015
                   2019-12-30      0.255819      0.219821                                           0.677052                           1.081693  127.763992              32.684456
                   2019-12-31      0.255819      0.219821                                           0.677052                           1.081693  127.462303              32.607277

        [244 rows x 6 columns]
        """
        self.check_same(data, except_data)

    def test_pref_operator(self):
        instruments = ["sh600519"]
        fields = [
            "PRef($$roewa_q, 201902)",
            "PRef($$yoyni_q, 201801)",
            "P($$roewa_q)",
            "P($$roewa_q) / PRef($$roewa_q, 201801)",
        ]
        data = D.features(instruments, fields, start_time="2018-04-28", end_time="2019-07-19", freq="day")
        except_data = """
                               PRef($$roewa_q, 201902)  PRef($$yoyni_q, 201801)  P($$roewa_q)  P($$roewa_q) / PRef($$roewa_q, 201801)
        instrument datetime
        sh600519   2018-05-02                      NaN                 0.395075      0.088887                                1.000000
                   2018-05-03                      NaN                 0.395075      0.088887                                1.000000
                   2018-05-04                      NaN                 0.395075      0.088887                                1.000000
                   2018-05-07                      NaN                 0.395075      0.088887                                1.000000
                   2018-05-08                      NaN                 0.395075      0.088887                                1.000000
        ...                                        ...                      ...           ...                                     ...
                   2019-07-15                 0.000000                 0.395075      0.000000                                0.000000
                   2019-07-16                 0.000000                 0.395075      0.000000                                0.000000
                   2019-07-17                 0.000000                 0.395075      0.000000                                0.000000
                   2019-07-18                 0.175322                 0.395075      0.175322                                1.972414
                   2019-07-19                 0.175322                 0.395075      0.175322                                1.972414

        [299 rows x 4 columns]
        """
        self.check_same(data, except_data)


if __name__ == "__main__":
    unittest.main()

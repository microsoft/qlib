# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import qlib
from qlib.data import D
import unittest


class TestPIT(unittest.TestCase):
    """
    NOTE!!!!!!
    The assert of this test assumes that users follows the cmd below and only download 2 stock.
    `python collector.py download_data --source_dir ./csv_pit --start 2000-01-01 --end 2020-01-01 --interval quarterly --symbol_flt_regx "^(600519|000725).*"`
    """

    def setUp(self):
        qlib.init(kernels=1)  # NOTE: set kernel to 1 to make it debug easier

    def to_str(self, obj):
        return "".join(str(obj).split())

    def check_same(self, a, b):
        self.assertEqual(self.to_str(a), self.to_str(b))

    def test_index_data(self):
        instruments = ["sh600519"]
        fields = ["$$roewa_q", "$$yoyni_q"]
        # Mao Tai published 2019Q2 report at 2019-07-13 & 2019-07-18
        # - http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search&lastPage=index
        data = D.features(instruments, fields, start_time="2019-01-01", end_time="20190719", freq="day")

        res = """
                $$roewa_q   $$yoyni_q
        count  133.000000  133.000000
        mean     0.196412    0.277930
        std      0.097591    0.030262
        min      0.000000    0.243892
        25%      0.094737    0.243892
        50%      0.255220    0.304181
        75%      0.255220    0.305041
        max      0.344644    0.305041
        """
        self.check_same(data.describe(), res)

        res = """
                               $$roewa_q  $$yoyni_q
        instrument datetime
        sh600519   2019-07-15   0.000000   0.305041
                   2019-07-16   0.000000   0.305041
                   2019-07-17   0.000000   0.305041
                   2019-07-18   0.175322   0.252650
                   2019-07-19   0.175322   0.252650
        """
        self.check_same(data.tail(), res)

    def test_no_exist_data(self):
        fields = ["$$roewa_q", "$$yoyni_q", "$close"]
        data = D.features(["sh600519", "sz000858"], fields, start_time="2019-01-01", end_time="20190719", freq="day")
        expect = """
                               $$roewa_q  $$yoyni_q      $close
        instrument datetime
        sh600519   2019-01-02    0.25522   0.243892  124.290070
                   2019-01-03    0.25522   0.243892  122.426697
                   2019-01-04    0.25522   0.243892  124.916748
                   2019-01-07    0.25522   0.243892  125.640930
                   2019-01-08    0.25522   0.243892  125.495667
        ...                          ...        ...         ...
        sz000858   2019-07-15        NaN        NaN   43.153912
                   2019-07-16        NaN        NaN   42.632988
                   2019-07-17        NaN        NaN   42.639885
                   2019-07-18        NaN        NaN   41.742931
                   2019-07-19        NaN        NaN   42.136211

        [266 rows x 3 columns]
        """
        self.check_same(data, expect)


if __name__ == "__main__":
    unittest.main()

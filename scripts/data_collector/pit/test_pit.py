# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import qlib
from qlib.data import D
import unittest


class TestPIT(unittest.TestCase):

    def setUp(self):
        qlib.init()

    def to_str(self, obj):
        return "".join(str(obj).split())

    def test_index_data(self):
        instruments = ["sh600519"]
        fields = ["$$roewa_q", "$$yoyni_q"]
        # Mao Tai published 2019Q2 report at 2019-07-13 & 2019-07-18
        # - http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search&lastPage=index
        data = D.features(instruments, fields, start_time="2019-01-01", end_time="20190719", freq="day")

        res = '''
                $$roewa_q   $$yoyni_q
        count  133.000000  133.000000
        mean     0.196412    0.277930
        std      0.097591    0.030262
        min      0.000000    0.243892
        25%      0.094737    0.243892
        50%      0.255220    0.304181
        75%      0.255220    0.305041
        max      0.344644    0.305041
        '''
        self.assertEqual(self.to_str(data.describe()), self.to_str(res))

        res = '''
                               $$roewa_q  $$yoyni_q
        instrument datetime
        sh600519   2019-07-15   0.000000   0.305041
                   2019-07-16   0.000000   0.305041
                   2019-07-17   0.000000   0.305041
                   2019-07-18   0.175322   0.252650
                   2019-07-19   0.175322   0.252650
        '''
        self.assertEqual(self.to_str(data.tail()), self.to_str(res))


if __name__ == "__main__":
    unittest.main()

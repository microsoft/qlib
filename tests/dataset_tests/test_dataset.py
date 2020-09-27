
import qlib
from qlib.data import D
from qlib.config import REG_CN
import unittest
import numpy as np


class TestDataset(unittest.TestCase):

    def setUp(self):
        provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
        qlib.init(provider_uri=provider_uri, region=REG_CN)

    def testCSI300(self):
        close_p = D.features(D.instruments('csi300'), ['$close'])
        size = close_p.groupby('datetime').size()
        cnt = close_p.groupby('datetime').count()
        size_desc = size.describe(percentiles=np.arange(0.1, 0.9, 0.1))
        cnt_desc = cnt.describe(percentiles=np.arange(0.1, 0.9, 0.1))

        print(size_desc)
        print(cnt_desc)

        self.assertLessEqual(size_desc.loc["max"][0], 305, "Excessive number of CSI300 constituent stocks")
        self.assertLessEqual(size_desc.loc["80%"][0], 290, "Insufficient number of CSI300 constituent stocks")
        
        self.assertLessEqual(cnt_desc.loc["max"][0], 305, "Excessive number of CSI300 constituent stocks")
        self.assertEqual(cnt_desc.loc["80%"][0], 300, "Insufficient number of CSI300 constituent stocks")

    def testClose(self):
        close_p = D.features(D.instruments('csi300'), ['Ref($close, 1)/$close - 1'])
        close_desc = close_p.describe(percentiles=np.arange(0.1, 0.9, 0.1))
        print(close_desc)
        self.assertLessEqual(abs(close_desc.loc["80%"][0]), 0.1, "Close value is abnormal")
        self.assertLessEqual(abs(close_desc.loc["max"][0]), 0.2, "Close value is abnormal")
        self.assertGreaterEqual(close_desc.loc["min"][0], -0.2, "Close value is abnormal")


if __name__ == '__main__':
    unittest.main()


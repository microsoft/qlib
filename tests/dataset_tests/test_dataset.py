
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

        print(size.describe(percentiles=np.arange(0.1, 0.9, 0.1)))
        print(cnt.describe(percentiles=np.arange(0.1, 0.9, 0.1)))
        # TODO: assert

    def testClose(self):
        close_p = D.features(D.instruments('csi300'), ['Ref($close, 1)/$close - 1'])
        print(close_p.describe(percentiles=np.arange(0.1, 0.9, 0.1)))
        # TODO: assert


if __name__ == '__main__':
    unittest.main()


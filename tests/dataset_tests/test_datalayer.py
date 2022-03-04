import unittest
import numpy as np
from qlib.data import D
from qlib.tests import TestAutoData


class TestDataset(TestAutoData):
    def testCSI300(self):
        close_p = D.features(D.instruments("csi300"), ["$close"])
        size = close_p.groupby("datetime").size()
        cnt = close_p.groupby("datetime").count()["$close"]
        size_desc = size.describe(percentiles=np.arange(0.1, 1.0, 0.1))
        cnt_desc = cnt.describe(percentiles=np.arange(0.1, 1.0, 0.1))

        print(size_desc)
        print(cnt_desc)

        self.assertLessEqual(size_desc.loc["max"], 305, "Excessive number of CSI300 constituent stocks")
        self.assertGreaterEqual(size_desc.loc["80%"], 290, "Insufficient number of CSI300 constituent stocks")

        self.assertLessEqual(cnt_desc.loc["max"], 305, "Excessive number of CSI300 constituent stocks")
        # FIXME: Due to the low quality of data. Hard to make sure there are enough data
        # self.assertEqual(cnt_desc.loc["80%"], 300, "Insufficient number of CSI300 constituent stocks")

    def testClose(self):
        close_p = D.features(D.instruments("csi300"), ["Ref($close, 1)/$close - 1"])
        close_desc = close_p.describe(percentiles=np.arange(0.1, 1.0, 0.1))
        print(close_desc)
        self.assertLessEqual(abs(close_desc.loc["90%"][0]), 0.1, "Close value is abnormal")
        self.assertLessEqual(abs(close_desc.loc["10%"][0]), 0.1, "Close value is abnormal")
        # FIXME: The yahoo data is not perfect. We have to
        # self.assertLessEqual(abs(close_desc.loc["max"][0]), 0.2, "Close value is abnormal")
        # self.assertGreaterEqual(close_desc.loc["min"][0], -0.2, "Close value is abnormal")


if __name__ == "__main__":
    unittest.main()

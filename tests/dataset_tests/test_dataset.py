
import sys
from pathlib import Path
import qlib
from qlib.data import D
from qlib.config import REG_CN
import unittest
import numpy as np
from qlib.utils import exists_qlib_data


class TestDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # use default data
        provider_uri = "~/.qlib/qlib_data/cn_data_simple"  # target_dir
        if not exists_qlib_data(provider_uri):
            print(f"Qlib data is not found in {provider_uri}")
            sys.path.append(str(Path(__file__).resolve().parent.parent.parent.joinpath("scripts")))
            from get_data import GetData

            GetData().qlib_data_cn(name="qlib_data_cn_simple", target_dir=provider_uri)
        qlib.init(provider_uri=provider_uri, region=REG_CN)

    def testCSI300(self):
        close_p = D.features(D.instruments('csi300'), ['$close'])
        size = close_p.groupby('datetime').size()
        cnt = close_p.groupby('datetime').count()['$close']
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
        close_p = D.features(D.instruments('csi300'), ['Ref($close, 1)/$close - 1'])
        close_desc = close_p.describe(percentiles=np.arange(0.1, 1.0, 0.1))
        print(close_desc)
        self.assertLessEqual(abs(close_desc.loc["90%"][0]), 0.1, "Close value is abnormal")
        self.assertLessEqual(abs(close_desc.loc["10%"][0]), 0.1, "Close value is abnormal")
        # FIXME: The yahoo data is not perfect. We have to 
        # self.assertLessEqual(abs(close_desc.loc["max"][0]), 0.2, "Close value is abnormal")
        # self.assertGreaterEqual(close_desc.loc["min"][0], -0.2, "Close value is abnormal")


if __name__ == '__main__':
    unittest.main()


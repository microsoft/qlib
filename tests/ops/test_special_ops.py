import unittest

from qlib.tests import TestOperatorData
from qlib.config import C
from qlib.data import D


class TestOperatorDataSetting(TestOperatorData):
    def test_setting(self):
        # All the query below passes
        df = D.features(["SH600519"], ["ChangeInstrument('SH000300', $close)"])

        # get market return for "SH600519"
        df = D.features(["SH600519"], ["ChangeInstrument('SH000300', Feature('close')/Ref(Feature('close'),1) -1)"])
        df = D.features(["SH600519"], ["ChangeInstrument('SH000300', $close/Ref($close,1) -1)"])
        # excess return
        df = D.features(["SH600519"], ["($close/Ref($close,1) -1) - ChangeInstrument('SH000300', $close/Ref($close,1) -1)"])
        print(df)


if __name__ == "__main__":
    unittest.main()

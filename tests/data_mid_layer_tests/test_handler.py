import os
import pickle
import shutil
import unittest
from qlib.tests import TestAutoData
from qlib.data import D
from qlib.data.dataset.handler import DataHandlerLP


class HandlerTests(TestAutoData):
    def to_str(self, obj):
        return "".join(str(obj).split())

    def test_handler_df(self):
        df = D.features(["sh600519"], start_time="20190101", end_time="20190201", fields=["$close"])
        dh = DataHandlerLP.from_df(df)
        print(dh.fetch())
        self.assertTrue(dh._data.equals(df))
        self.assertTrue(dh._infer is dh._data)
        self.assertTrue(dh._learn is dh._data)
        self.assertTrue(dh.data_loader._data is dh._data)
        fname = "_handler_test.pkl"
        dh.to_pickle(fname, dump_all=True)

        with open(fname, "rb") as f:
            dh_d = pickle.load(f)

        self.assertTrue(dh_d._data.equals(df))
        self.assertTrue(dh_d._infer is dh_d._data)
        self.assertTrue(dh_d._learn is dh_d._data)
        # Data loader will no longer be useful
        self.assertTrue("_data" not in dh_d.data_loader.__dict__.keys())
        os.remove(fname)


if __name__ == "__main__":
    unittest.main()

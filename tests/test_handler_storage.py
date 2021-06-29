import unittest
import qlib
import time
import pandas as pd

from qlib.data import D
from qlib.tests import TestAutoData

from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import Processor
from qlib.contrib.data.handler import check_transform_proc
from qlib.utils import init_instance_by_config
from qlib.log import TimeInspector


class TestHandler(DataHandlerLP):
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        infer_processors=[],
        learn_processors=[],
        fit_start_time=None,
        fit_end_time=None,
        drop_raw=True,
    ):

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "freq": "day",
                "config": self.get_feature_config(),
                "swap_level": False,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            drop_raw=drop_raw,
        )

    def get_feature_config(self):
        fields = ["Ref($open, 1)", "Ref($close, 1)", "Ref($volume, 1)", "$open", "$close", "$volume"]
        names = ["open_0", "close_0", "volume_0", "open_1", "close_1", "volume_1"]
        return fields, names


class MiniTimer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        print(f"[MyTimer Info] <{self.name}> process costs {self.end - self.start} seconds")


class TestHandlerStorage(TestAutoData):

    market = "all"

    start_time = "2020-01-01"
    end_time = "2020-12-31"
    train_end_time = "2020-05-31"
    test_start_time = "2020-06-01"

    data_handler_kwargs = {
        "start_time": start_time,
        "end_time": end_time,
        "fit_start_time": start_time,
        "fit_end_time": train_end_time,
        "instruments": market,
        "infer_processors": ["HashingStock"],
    }

    def test_handler_storage(self):
        with MiniTimer("init data hanlder"):
            data_handler = TestHandler(**self.data_handler_kwargs)

        with MiniTimer("random fetch"):
            print(data_handler.fetch(selector=("SH600170", slice(None)), level=None))
            print(
                data_handler.fetch(
                    selector=("SH600170", slice(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"))), level=None
                )
            )
            print(
                data_handler.fetch(
                    selector=(["SH600170", "SH600383"], slice(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"))),
                    level=None,
                )
            )


if __name__ == "__main__":
    unittest.main()

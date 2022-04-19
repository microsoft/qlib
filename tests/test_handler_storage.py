import unittest
import time
import numpy as np
from qlib.data import D
from qlib.tests import TestAutoData

from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import check_transform_proc
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


class TestHandlerStorage(TestAutoData):

    market = "all"

    start_time = "2010-01-01"
    end_time = "2020-12-31"
    train_end_time = "2015-12-31"
    test_start_time = "2016-01-01"

    data_handler_kwargs = {
        "start_time": start_time,
        "end_time": end_time,
        "fit_start_time": start_time,
        "fit_end_time": train_end_time,
        "instruments": market,
    }

    def test_handler_storage(self):
        # init data handler
        data_handler = TestHandler(**self.data_handler_kwargs)

        # init data handler with hasing storage
        data_handler_hs = TestHandler(**self.data_handler_kwargs, infer_processors=["HashStockFormat"])

        fetch_start_time = "2019-01-01"
        fetch_end_time = "2019-12-31"
        instruments = D.instruments(market=self.market)
        instruments = D.list_instruments(
            instruments=instruments, start_time=fetch_start_time, end_time=fetch_end_time, as_list=True
        )

        with TimeInspector.logt("random fetch with DataFrame Storage"):

            # single stock
            for i in range(100):
                random_index = np.random.randint(len(instruments), size=1)[0]
                fetch_stock = instruments[random_index]
                data_handler.fetch(selector=(fetch_stock, slice(fetch_start_time, fetch_end_time)), level=None)

            # multi stocks
            for i in range(100):
                random_indexs = np.random.randint(len(instruments), size=5)
                fetch_stocks = [instruments[_index] for _index in random_indexs]
                data_handler.fetch(selector=(fetch_stocks, slice(fetch_start_time, fetch_end_time)), level=None)

        with TimeInspector.logt("random fetch with HashingStock Storage"):

            # single stock
            for i in range(100):
                random_index = np.random.randint(len(instruments), size=1)[0]
                fetch_stock = instruments[random_index]
                data_handler_hs.fetch(selector=(fetch_stock, slice(fetch_start_time, fetch_end_time)), level=None)

            # multi stocks
            for i in range(100):
                random_indexs = np.random.randint(len(instruments), size=5)
                fetch_stocks = [instruments[_index] for _index in random_indexs]
                data_handler_hs.fetch(selector=(fetch_stocks, slice(fetch_start_time, fetch_end_time)), level=None)


if __name__ == "__main__":
    unittest.main()

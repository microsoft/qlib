#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import qlib
import fire
import pickle
import pandas as pd

from datetime import datetime
from qlib.config import REG_CN
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import Alpha158
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.tests.data import GetData

class RollingDataWorkflow(object):

    MARKET = "csi300"
    start_time = "2010-01-01"
    end_time = "2019-12-31" 
    rolling_cnt = 5

    def _init_qlib(self):
        """initialize qlib"""
        # use yahoo_cn_1min data
        provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
        if not exists_qlib_data(provider_uri):
            print(f"Qlib data is not found in {provider_uri}")
            GetData().qlib_data(target_dir=provider_uri, region=REG_CN)
        qlib.init(provider_uri=provider_uri, region=REG_CN)
    
    def _dump_pre_handler(self, path):
        handler_config = {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "instruments": self.MARKET,
            },
        }
        pre_handler = init_instance_by_config(handler_config)
        pre_handler.to_pickle(path)

    def _load_pre_handler(self, path):
        with open(path, "rb") as file_dataset:
            pre_handler = pickle.load(file_dataset)
        return pre_handler

    def rolling_process(self):
        self._init_qlib()
        self._dump_pre_handler("pre_handler.py")
        pre_handler = self._load_pre_handler("pre_handler.py")

        train_start_time = (2010,1,1)
        train_end_time = (2012,12,31)
        valid_start_time = (2013,1,1)
        valid_end_time = (2013,12,31)
        test_start_time = (2014,1,1)
        test_end_time = (2014,12,31)
        
        dataset_config = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "RollingDataHandler",
                    "module_path": "rolling_handler",
                    "kwargs": {
                        "start_time": datetime(*train_start_time),
                        "end_time": datetime(*test_end_time),
                        "fit_start_time": datetime(*train_start_time),
                        "fit_end_time": datetime(*train_end_time),
                        "data_loader_kwargs":{
                            "handler_config": pre_handler,
                        }
                    },
                },
                "segments": {
                    "train": (datetime(*train_start_time), datetime(*train_end_time)),
                    "valid": (datetime(*valid_start_time), datetime(*valid_end_time)),
                    "test": (datetime(*test_start_time), datetime(*test_end_time)),
                },
            },
        }

        dataset = init_instance_by_config(dataset_config)

        for rolling_offset in range(rolling_cnt):
            if rolling_offset:
                dataset.init(
                    handler_kwargs={
                        "init_type": DataHandlerLP.IT_FIT_SEQ,
                        "start_time": datetime(train_start_time[0] + 1, *train_start_time[1:]),
                        "end_time": datetime(test_end_time[0] + 1, *test_end_time[1:]),
                    },
                    segment_kwargs={
                        "train": (datetime(train_start_time[0] + 1, *train_start_time[1:]), datetime(train_end_time[0], *train_end_time[1:])),
                        "valid": (datetime(valid_start_time[0] + 1, *valid_start_time[1:]), datetime(valid_end_time[0], *valid_end_time[1:])),
                        "test": (datetime(test_start_time[0] + 1, *test_start_time[1:]), datetime(test_end_time[0], *test_end_time[1:])),
                    },
                )

            dtrain, dvalid, dtest  = dataset.prepare(["train", "valid", "test"])
            

if __name__ == "__main__":
    fire.Fire(RollingDataWorkflow)

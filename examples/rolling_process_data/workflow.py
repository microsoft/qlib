#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import qlib
import pickle
import datetime
import pandas as pd
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
                "start_time": start_time,
                "end_time": end_time,
                "instruments": MARKET,
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

        init_start_time = datetime.datetime(2010,1,1)
        init_end_time = datetime.datetime(2014,12,31)
        init_fit_end_time = datetime.datetime(2012,12,31)

        dataset_config = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "RollingDataHandler",
                    "module_path": "rolling_handler",
                    "kwargs": {
                        "start_time": init_start_time,
                        "end_time": init_start_time,
                        "fit_start_time": init_fit_start_time,
                        "fit_end_time": init_fit_end_time,
                        "data_loader_kwargs":{
                            "handler_config": pre_handler,
                        }
                    },
                },
                "segments": {
                    "train": (init_start_time, init_fit_end_time),
                    "valid": (init_start_time, "2013-12-31"),
                    "test": (init_start_time, init_end_time),
                },
            },
        }

        dataset = init_instance_by_config(dataset_config)

        for rolling_offset in range(rolling_cnt):
            if rolling_offset:
                dataset.init(
                    handler_kwargs={
                        "init_type": DataHandlerLP.IT_FIT_IND,
                        "start_time": "2021-01-19 00:00:00",
                        "end_time": "2021-01-25 16:00:00",
                    },
                    segment_kwargs={
                        "train": ("2010-01-01", "2012-12-31"),
                        "valid": ("2013-01-01", "2013-12-31"),
                        "test": ("2014-01-01", "2014-12-31"),
                    },
                )


if __name__ == "__main__":

    # use default data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    if not exists_qlib_data(provider_uri):
        print(f"Qlib data is not found in {provider_uri}")
        GetData().qlib_data(target_dir=provider_uri, region=REG_CN)

    qlib.init(provider_uri=provider_uri, region=REG_CN)

    market = "csi300"
    benchmark = "SH000300"

    ###################################
    # train model
    ###################################
    data_handler_config = {
        "start_time": "2008-01-01",
        "end_time": "2020-08-01",
        "fit_start_time": "2008-01-01",
        "fit_end_time": "2014-12-31",
        "instruments": market,
    }

    task = {
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "train": ("2008-01-01", "2014-12-31"),
                    "valid": ("2015-01-01", "2016-12-31"),
                    "test": ("2017-01-01", "2020-08-01"),
                },
            },
        },
    }

    dataset = init_instance_by_config(task["dataset"])


#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import sys
from pathlib import Path

import qlib
import pickle
import numpy as np
import pandas as pd
from qlib.config import REG_CN
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.strategy.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import (
    backtest as normal_backtest,
    risk_analysis,
)

from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.ops import Operators
from qlib.data.data import Cal

from highfreq_ops import DayFirst, DayLast, FFillNan, Date, Select, IsNull

if __name__ == "__main__":

    # use default data
    provider_uri = "/mnt/v-xiabi/data/qlib/high_freq"  # target_dir
    qlib.init(
        provider_uri=provider_uri,
        custom_ops=[DayFirst, DayLast, FFillNan, Date, Select, IsNull],
        redis_port=-1,
        region=REG_CN,
        auto_mount=False,
    )

    MARKET = "all"
    BENCHMARK = "SH000300"

    start_time = "2019-01-01 00:00:00"
    end_time = "2019-12-31 15:00:00"
    train_end_time = "2019-05-31 15:00:00"
    test_start_time = "2019-06-01 00:00:00"

    ###################################
    # train model
    ###################################
    DATA_HANDLER_CONFIG0 = {
        "start_time": start_time,
        "end_time": end_time,
        "freq": "1min",
        "fit_start_time": start_time,
        "fit_end_time": train_end_time,
        "instruments": MARKET,
        "infer_processors": [{"class": "HighFreqNorm", "module_path": "highfreq_processor", "kwargs": {}}],
    }
    DATA_HANDLER_CONFIG1 = {
        "start_time": start_time,
        "end_time": end_time,
        "freq": "1min",
        "instruments": MARKET,
    }

    task = {
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "HighFreqHandler",
                    "module_path": "highfreq_handler",
                    "kwargs": DATA_HANDLER_CONFIG0,
                },
                "segments": {
                    "train": (start_time, train_end_time),
                    "test": (
                        test_start_time,
                        end_time,
                    ),
                },
            },
        },
        # You shoud record the data in specific sequence
        # "record": ['SignalRecord', 'SigAnaRecord', 'PortAnaRecord'],
        "dataset_backtest": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "HighFreqBacktestHandler",
                    "module_path": "highfreq_handler",
                    "kwargs": DATA_HANDLER_CONFIG1,
                },
                "segments": {
                    "train": (start_time, train_end_time),
                    "test": (
                        test_start_time,
                        end_time,
                    ),
                },
            },
        },
    }
    ##=============load the calendar for cache=============
    Cal.calendar(freq="1min")
    Cal.get_calendar_day(freq="1min")

    ##=============get data=============
    dataset = init_instance_by_config(task["dataset"])
    dataset_backtest = init_instance_by_config(task["dataset_backtest"])
    xtrain, xtest = dataset.prepare(["train", "test"])
    backtest_train, backtest_test = dataset_backtest.prepare(["train", "test"])
    print(xtrain, xtest)
    print(backtest_train, backtest_test)
    del xtrain, xtest
    del backtest_train, backtest_test

    ##=============dump dataset=============
    dataset.to_pickle(path="dataset.pkl")
    dataset_backtest.to_pickle(path="dataset_backtest.pkl")

    del dataset, dataset_backtest
    ##=============reload dataset=============
    file_dataset = open("dataset.pkl", "rb")
    dataset = pickle.load(file_dataset)
    file_dataset.close()

    file_dataset_backtest = open("dataset_backtest.pkl", "rb")
    dataset_backtest = pickle.load(file_dataset_backtest)

    file_dataset_backtest.close()

    ##=============reload_dataset=============
    dataset.init(init_type=DataHandlerLP.IT_LS)
    dataset_backtest.init(init_type=DataHandlerLP.IT_LS)

    ##=============reinit qlib=============
    qlib.init(
        provider_uri=provider_uri,
        custom_ops=[DayFirst, DayLast, FFillNan, Date, Select, IsNull],
        redis_port=-1,
        region=REG_CN,
        auto_mount=False,
    )

    Cal.calendar(freq="1min")  # load the calendar for cache
    Cal.get_calendar_day(freq="1min")  # load the calendar for cache

    ##=============test dataset
    xtrain, xtest = dataset.prepare(["train", "test"])
    backtest_train, backtest_test = dataset_backtest.prepare(["train", "test"])

    print(xtrain, xtest)
    print(backtest_train, backtest_test)
    del xtrain, xtest
    del backtest_train, backtest_test

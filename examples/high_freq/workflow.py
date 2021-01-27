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
from qlib.utils import exists_qlib_data

from highfreq_ops import DayFirst, DayLast, FFillNan, Date, Select, IsNull

if __name__ == "__main__":

    # use yahoo_cn_1min data
    provider_uri = "~/.qlib/qlib_data/yahoo_cn_1min"
    if not exists_qlib_data(provider_uri):
        print(f"Qlib data is not found in {provider_uri}")
        sys.path.append(str(Path(__file__).resolve().parent.parent.parent.joinpath("scripts")))
        from get_data import GetData

        GetData().qlib_data(target_dir=provider_uri, interval="1min", region=REG_CN)

    qlib.init(
        provider_uri=provider_uri,
        custom_ops=[DayFirst, DayLast, FFillNan, Date, Select, IsNull],
        redis_port=-1,
        region=REG_CN,
        auto_mount=False,
    )

    MARKET = "all"
    BENCHMARK = "SH000300"
    DROP_LOAD_DATASET = False  # flag wether to test [drop and load dataset]

    # start_time = "2019-01-01 00:00:00"
    # end_time = "2019-12-31 15:00:00"
    # train_end_time = "2019-05-31 15:00:00"
    # test_start_time = "2019-06-01 00:00:00"
    start_time = "2020-09-14 00:00:00"
    end_time = "2021-01-18 16:00:00"
    train_end_time = "2020-11-30 16:00:00"
    test_start_time = "2020-12-01 00:00:00"
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
    # unnecessary, but may accelerate
    Cal.calendar(freq="1min")  # load the calendar for cache
    Cal.get_calendar_day(freq="1min")  # load the calendar for cache

    ##=============get data=============

    dataset = init_instance_by_config(task["dataset"])
    xtrain, xtest = dataset.prepare(["train", "test"])
    print(xtrain, xtest)

    dataset_backtest = init_instance_by_config(task["dataset_backtest"])
    backtest_train, backtest_test = dataset_backtest.prepare(["train", "test"])
    print(backtest_train, backtest_test)

    del xtrain, xtest
    del backtest_train, backtest_test

    ## example to show how to save the dataset and reload it, and how to use different data
    if DROP_LOAD_DATASET:

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
        ## Unless you want to modify the provider_uri and other configurations, reinit is unnecessary
        qlib.init(
            provider_uri=provider_uri,
            custom_ops=[DayFirst, DayLast, FFillNan, Date, Select, IsNull],
            redis_port=-1,
            region=REG_CN,
            auto_mount=False,
        )

        Cal.calendar(freq="1min")  # load the calendar for cache
        Cal.get_calendar_day(freq="1min")  # load the calendar for cache

        ##=============test dataset=============
        xtrain, xtest = dataset.prepare(["train", "test"])
        backtest_train, backtest_test = dataset_backtest.prepare(["train", "test"])

        print(xtrain, xtest)
        print(backtest_train, backtest_test)
        del xtrain, xtest
        del backtest_train, backtest_test

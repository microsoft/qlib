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


def save_dataset(dataset, path: [Path, str]):
    """
    save dataset to path

    Parameters
    ----------
    path : [Path, str]
        path to save
    """
    dataset.to_pickle(path=path)


def load_dataset(path: [Path, str], init_type=DataHandlerLP.IT_LS):
    """
    load dataset from path

    Parameters
    ----------
    path : [Path, str]
        path to load

    init_type : str
        - if `init_type` == DataHandlerLP.IT_FIT_SEQ:

            the input of `DataHandlerLP.fit` will be the output of the previous processor

        - if `init_type` == DataHandlerLP.IT_FIT_IND:

            the input of `DataHandlerLP.fit` will be the original df

        - if `init_type` == DataHandlerLP.IT_LS:

            The state of the object has been load by pickle
    """
    fd = open(path, "rb")
    dataset = pickle.load(fd)
    dataset.init(init_type=init_type)
    fd.close()
    return dataset


if __name__ == "__main__":

    # use default data
    provider_uri = "/mnt/v-xiabi/data/qlib/high_freq"  # target_dir
    qlib.init(
        provider_uri=provider_uri,
        custom_ops=[DayFirst, DayLast, FFillNan, Date, Select, IsNull],
        redis_port=233,
        region=REG_CN,
        auto_mount=False,
    )

    MARKET = "csi300"
    BENCHMARK = "SH000300"

    ###################################
    # train model
    ###################################
    DATA_HANDLER_CONFIG0 = {
        "start_time": "2017-01-01 00:00:00",
        "end_time": "2020-11-30 15:00:00",
        "freq": "1min",
        "fit_start_time": "2017-01-01 00:00:00",
        "fit_end_time": "2020-08-31 15:00:00",
        "instruments": "all",
        "infer_processors": [{"class": "HighFreqNorm", "module_path": "highfreq_processor", "kwargs": {}}],
    }
    DATA_HANDLER_CONFIG1 = {
        "start_time": "2017-01-01 00:00:00",
        "end_time": "2020-11-30 15:00:00",
        "freq": "1min",
        "instruments": "all",
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
                    "train": ("2017-01-01 00:00:00", "2020-08-31 15:00:00"),
                    "test": (
                        "2020-09-01 00:00:00",
                        "2020-11-30 15:00:00",
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
                    "module_path": "highfreq_hander",
                    "kwargs": DATA_HANDLER_CONFIG1,
                },
                "segments": {
                    "train": ("2017-01-01 00:00:00", "2020-08-31 15:00:00"),
                    "test": (
                        "2020-09-01 00:00:00",
                        "2020-11-30 15:00:00",
                    ),
                },
            },
        },
    }
    Cal.get_calender_day(freq="1min")  # TO FIX: load the calendar day for cache
    dataset = init_instance_by_config(task["dataset"])
    dataset_backtest = init_instance_by_config(task["dataset_backtest"])

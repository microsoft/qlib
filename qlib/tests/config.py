#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

CSI300_MARKET = "csi300"
CSI100_MARKET = "csi100"

CSI300_BENCH = "SH000300"

DATASET_ALPHA158_CLASS = "Alpha158"
DATASET_ALPHA360_CLASS = "Alpha360"

###################################
# config
###################################


GBDT_MODEL = {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
        "loss": "mse",
        "colsample_bytree": 0.8879,
        "learning_rate": 0.0421,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "num_threads": 20,
    },
}


RECORD_CONFIG = [
    {
        "class": "SignalRecord",
        "module_path": "qlib.workflow.record_temp",
    },
    {
        "class": "SigAnaRecord",
        "module_path": "qlib.workflow.record_temp",
    },
]


def get_data_handler_config(market=CSI300_MARKET):
    return {
        "start_time": "2008-01-01",
        "end_time": "2020-08-01",
        "fit_start_time": "2008-01-01",
        "fit_end_time": "2014-12-31",
        "instruments": market,
    }


def get_dataset_config(market=CSI300_MARKET, dataset_class=DATASET_ALPHA158_CLASS):
    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": dataset_class,
                "module_path": "qlib.contrib.data.handler",
                "kwargs": get_data_handler_config(market),
            },
            "segments": {
                "train": ("2008-01-01", "2014-12-31"),
                "valid": ("2015-01-01", "2016-12-31"),
                "test": ("2017-01-01", "2020-08-01"),
            },
        },
    }


def get_gbdt_task(market=CSI300_MARKET):
    return {
        "model": GBDT_MODEL,
        "dataset": get_dataset_config(market),
    }


def get_record_lgb_config(market=CSI300_MARKET):
    return {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
        },
        "dataset": get_dataset_config(market),
        "record": RECORD_CONFIG,
    }


def get_record_xgboost_config(market=CSI300_MARKET):
    return {
        "model": {
            "class": "XGBModel",
            "module_path": "qlib.contrib.model.xgboost",
        },
        "dataset": get_dataset_config(market),
        "record": RECORD_CONFIG,
    }


CSI300_DATASET_CONFIG = get_dataset_config(market=CSI300_MARKET)
CSI300_GBDT_TASK = get_gbdt_task(market=CSI300_MARKET)

CSI100_RECORD_XGBOOST_TASK_CONFIG = get_record_xgboost_config(market=CSI100_MARKET)
CSI100_RECORD_LGB_TASK_CONFIG = get_record_lgb_config(market=CSI100_MARKET)

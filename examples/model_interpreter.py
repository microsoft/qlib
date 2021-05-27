#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.


import qlib
from qlib.config import REG_CN

from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.tests.data import GetData

market = "csi300"
benchmark = "SH000300"

###################################
# config
###################################
data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": market,
}

task = {
    "model": {
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
    },
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


if __name__ == "__main__":

    # use default data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    if not exists_qlib_data(provider_uri):
        print(f"Qlib data is not found in {provider_uri}")
        GetData().qlib_data(target_dir=provider_uri, region=REG_CN)

    qlib.init(provider_uri=provider_uri, region=REG_CN)

    ###################################
    # train model
    ###################################
    # model initialization
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])
    model.fit(dataset)

    # get model feature importance
    feature_importance = model.get_feature_importance()
    print("feature importance:")
    print(feature_importance)

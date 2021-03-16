import qlib
from qlib.model.trainer import task_train
from qlib.workflow.task.online import OnlineManager
from qlib.config import REG_CN
import fire
from qlib.workflow import R

data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": "csi100",
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
    "record": {
        "class": "SignalRecord",
        "module_path": "qlib.workflow.record_temp",
    },
}


def first_train(experiment_name="online_svr"):

    rid = task_train(task_config=task, experiment_name=experiment_name)

    rom = OnlineManager(experiment_name)
    rom.reset_online_model(rid)


def update_online_pred(experiment_name="online_svr"):

    rom = OnlineManager(experiment_name)

    print("Here are the online models waiting for update:")
    for rid, rec in rom.list_online_model().items():
        print(rid)

    rom.update_online_pred()


if __name__ == "__main__":
    ## to train a model and set it to online model, use the command below
    # python update_online_pred.py first_train
    ## to update online predictions once a day, use the command below
    # python update_online_pred.py update_online_pred

    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    fire.Fire()

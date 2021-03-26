import fire
import qlib
from qlib.config import REG_CN
from qlib.model.trainer import task_train
from qlib.workflow.task.online import OnlineManagerR
from qlib.workflow.task.utils import list_recorders

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

    online_manager = OnlineManagerR(experiment_name)
    online_manager.reset_online_tag(rid)


def update_online_pred(experiment_name="online_svr"):

    online_manager = OnlineManagerR(experiment_name)

    print("Here are the online models waiting for update:")
    for rid, rec in list_recorders(experiment_name).items():
        if online_manager.get_online_tag(rec) == OnlineManagerR.ONLINE_TAG:
            print(rid)

    online_manager.update_online_pred()


if __name__ == "__main__":
    ## to train a model and set it to online model, use the command below
    # python update_online_pred.py first_train
    ## to update online predictions once a day, use the command below
    # python update_online_pred.py update_online_pred

    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    fire.Fire()

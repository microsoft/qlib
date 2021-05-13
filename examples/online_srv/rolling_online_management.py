# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This example show how OnlineManager works with rolling tasks.
There are two parts including first train and routine.
Firstly, the OnlineManager will finish the first training and set trained models to `online` models.
Next, the OnlineManager will finish a routine process, including update online prediction -> prepare signals -> prepare tasks -> prepare new models -> reset online models
"""

import os
from pathlib import Path
import pickle
import fire
import qlib
from qlib.workflow import R
from qlib.workflow.online.strategy import RollingStrategy
from qlib.workflow.task.gen import RollingGen
from qlib.workflow.task.manage import TaskManager
from qlib.workflow.online.manager import OnlineManager

data_handler_config = {
    "start_time": "2013-01-01",
    "end_time": "2020-09-25",
    "fit_start_time": "2013-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": "csi100",
}

dataset_config = {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": data_handler_config,
        },
        "segments": {
            "train": ("2013-01-01", "2014-12-31"),
            "valid": ("2015-01-01", "2015-12-31"),
            "test": ("2016-01-01", "2020-07-10"),
        },
    },
}

record_config = [
    {
        "class": "SignalRecord",
        "module_path": "qlib.workflow.record_temp",
    },
    {
        "class": "SigAnaRecord",
        "module_path": "qlib.workflow.record_temp",
    },
]

# use lgb model
task_lgb_config = {
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
    },
    "dataset": dataset_config,
    "record": record_config,
}

# use xgboost model
task_xgboost_config = {
    "model": {
        "class": "XGBModel",
        "module_path": "qlib.contrib.model.xgboost",
    },
    "dataset": dataset_config,
    "record": record_config,
}


class RollingOnlineExample:
    def __init__(
        self,
        provider_uri="~/.qlib/qlib_data/cn_data",
        region="cn",
        task_url="mongodb://10.0.0.4:27017/",
        task_db_name="rolling_db",
        rolling_step=550,
        tasks=[task_xgboost_config, task_lgb_config],
    ):
        mongo_conf = {
            "task_url": task_url,  # your MongoDB url
            "task_db_name": task_db_name,  # database name
        }
        qlib.init(provider_uri=provider_uri, region=region, mongo=mongo_conf)
        self.tasks = tasks
        self.rolling_step = rolling_step
        strategy = []
        for task in tasks:
            name_id = task["model"]["class"]  # NOTE: Assumption: The model class can specify only one strategy
            strategy.append(
                RollingStrategy(
                    name_id,
                    task,
                    RollingGen(step=rolling_step, rtype=RollingGen.ROLL_SD),
                )
            )

        self.rolling_online_manager = OnlineManager(strategy)
        self.collector = self.rolling_online_manager.get_collector()

    _ROLLING_MANAGER_PATH = (
        ".RollingOnlineExample"  # the OnlineManager will dump to this file, for it can be loaded when calling routine.
    )

    # Reset all things to the first status, be careful to save important data
    def reset(self):
        for task in self.tasks:
            name_id = task["model"]["class"]
            TaskManager(name_id).remove()
            exp = R.get_exp(experiment_name=name_id)
            for rid in exp.list_recorders():
                exp.delete_recorder(rid)

            if os.path.exists(self._ROLLING_MANAGER_PATH):
                os.remove(self._ROLLING_MANAGER_PATH)

    def first_run(self):
        print("========== reset ==========")
        self.reset()
        print("========== first_run ==========")
        self.rolling_online_manager.first_train()
        print("========== dump ==========")
        self.rolling_online_manager.to_pickle(self._ROLLING_MANAGER_PATH)
        print("========== collect results ==========")
        print(self.collector())

    def routine(self):
        print("========== load ==========")
        with Path(self._ROLLING_MANAGER_PATH).open("rb") as f:
            self.rolling_online_manager = pickle.load(f)
        print("========== routine ==========")
        self.rolling_online_manager.routine()
        print("========== collect results ==========")
        print(self.collector())
        print("========== signals ==========")
        print(self.rolling_online_manager.get_signals())

    def main(self):
        self.first_run()
        self.routine()


if __name__ == "__main__":
    ####### to train the first version's models, use the command below
    # python rolling_online_management.py first_run

    ####### to update the models and predictions after the trading time, use the command below
    # python rolling_online_management.py after_day

    ####### to define your own parameters, use `--`
    # python rolling_online_management.py first_run --exp_name='your_exp_name' --rolling_step=40
    fire.Fire(RollingOnlineExample)

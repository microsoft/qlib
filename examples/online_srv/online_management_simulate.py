# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This example is about how can simulate the OnlineManager based on rolling tasks. 
"""

import fire
import qlib
from qlib.model.trainer import DelayTrainerR, DelayTrainerRM, TrainerR, TrainerRM
from qlib.workflow import R
from qlib.workflow.online.manager import OnlineManager
from qlib.workflow.online.strategy import RollingStrategy
from qlib.workflow.task.gen import RollingGen
from qlib.workflow.task.manage import TaskManager


data_handler_config = {
    "start_time": "2018-01-01",
    "end_time": "2018-10-31",
    "fit_start_time": "2018-01-01",
    "fit_end_time": "2018-03-31",
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
            "train": ("2018-01-01", "2018-03-31"),
            "valid": ("2018-04-01", "2018-05-31"),
            "test": ("2018-06-01", "2018-09-10"),
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


class OnlineSimulationExample:
    def __init__(
        self,
        provider_uri="~/.qlib/qlib_data/cn_data",
        region="cn",
        exp_name="rolling_exp",
        task_url="mongodb://10.0.0.4:27017/",
        task_db_name="rolling_db",
        task_pool="rolling_task",
        rolling_step=80,
        start_time="2018-09-10",
        end_time="2018-10-31",
        tasks=[task_xgboost_config, task_lgb_config],
    ):
        """
        Init OnlineManagerExample.

        Args:
            provider_uri (str, optional): the provider uri. Defaults to "~/.qlib/qlib_data/cn_data".
            region (str, optional): the stock region. Defaults to "cn".
            exp_name (str, optional): the experiment name. Defaults to "rolling_exp".
            task_url (str, optional): your MongoDB url. Defaults to "mongodb://10.0.0.4:27017/".
            task_db_name (str, optional): database name. Defaults to "rolling_db".
            task_pool (str, optional): the task pool name (a task pool is a collection in MongoDB). Defaults to "rolling_task".
            rolling_step (int, optional): the step for rolling. Defaults to 80.
            start_time (str, optional): the start time of simulating. Defaults to "2018-09-10".
            end_time (str, optional): the end time of simulating. Defaults to "2018-10-31".
            tasks (dict or list[dict]): a set of the task config waiting for rolling and training
        """
        self.exp_name = exp_name
        self.task_pool = task_pool
        self.start_time = start_time
        self.end_time = end_time
        mongo_conf = {
            "task_url": task_url,
            "task_db_name": task_db_name,
        }
        qlib.init(provider_uri=provider_uri, region=region, mongo=mongo_conf)
        self.rolling_gen = RollingGen(
            step=rolling_step, rtype=RollingGen.ROLL_SD, ds_extra_mod_func=None
        )  # The rolling tasks generator, ds_extra_mod_func is None because we just need to simulate to 2018-10-31 and needn't change the handler end time.
        self.trainer = DelayTrainerRM(self.exp_name, self.task_pool)  # Also can be TrainerR, TrainerRM, DelayTrainerR
        self.rolling_online_manager = OnlineManager(
            RollingStrategy(exp_name, task_template=tasks, rolling_gen=self.rolling_gen),
            trainer=self.trainer,
            begin_time=self.start_time,
        )
        self.tasks = tasks

    # Reset all things to the first status, be careful to save important data
    def reset(self):
        TaskManager(self.task_pool).remove()
        exp = R.get_exp(experiment_name=self.exp_name)
        for rid in exp.list_recorders():
            exp.delete_recorder(rid)

    # Run this to run all workflow automatically
    def main(self):
        print("========== reset ==========")
        self.reset()
        print("========== simulate ==========")
        self.rolling_online_manager.simulate(end_time=self.end_time)
        print("========== collect results ==========")
        print(self.rolling_online_manager.get_collector()())
        print("========== signals ==========")
        print(self.rolling_online_manager.get_signals())


if __name__ == "__main__":
    ## to run all workflow automatically with your own parameters, use the command below
    # python online_management_simulate.py main --experiment_name="your_exp_name" --rolling_step=60
    fire.Fire(OnlineSimulationExample)

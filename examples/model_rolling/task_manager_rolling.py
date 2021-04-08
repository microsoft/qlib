from pprint import pprint

import fire
import qlib
from qlib.config import REG_CN
from qlib.model.trainer import task_train
from qlib.workflow import R
from qlib.workflow.task.gen import RollingGen, task_generator
from qlib.workflow.task.manage import TaskManager, run_task
from qlib.workflow.task.collect import RecorderCollector
from qlib.model.ens.ensemble import RollingEnsemble, ens_workflow
import pandas as pd
from qlib.workflow.task.utils import list_recorders
from qlib.model.ens.group import RollingGroup
from qlib.model.trainer import TrainerRM

data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
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
            "train": ("2008-01-01", "2014-12-31"),
            "valid": ("2015-01-01", "2016-12-31"),
            "test": ("2017-01-01", "2020-08-01"),
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

# use lgb
task_lgb_config = {
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
    },
    "dataset": dataset_config,
    "record": record_config,
}

# use xgboost
task_xgboost_config = {
    "model": {
        "class": "XGBModel",
        "module_path": "qlib.contrib.model.xgboost",
    },
    "dataset": dataset_config,
    "record": record_config,
}

# Reset all things to the first status, be careful to save important data
def reset(task_pool, exp_name):
    print("========== reset ==========")
    TaskManager(task_pool=task_pool).remove()

    exp = R.get_exp(experiment_name=exp_name)

    for rid in exp.list_recorders():
        exp.delete_recorder(rid)


# This part corresponds to "Task Generating" in the document
def task_generating():

    print("========== task_generating ==========")

    tasks = task_generator(
        tasks=[task_xgboost_config, task_lgb_config],
        generators=RollingGen(step=550, rtype=RollingGen.ROLL_SD),  # generate different date segment
    )

    pprint(tasks)

    return tasks


def task_training(tasks, task_pool, exp_name):
    trainer = TrainerRM(exp_name, task_pool)
    trainer.train(tasks)


# This part corresponds to "Task Collecting" in the document
def task_collecting(task_pool, exp_name):
    print("========== task_collecting ==========")

    def rec_key(recorder):
        task_config = recorder.load_object("task")
        model_key = task_config["model"]["class"]
        rolling_key = task_config["dataset"]["kwargs"]["segments"]["test"]
        return model_key, rolling_key

    def my_filter(recorder):
        # only choose the results of "LGBModel"
        model_key, rolling_key = rec_key(recorder)
        if model_key == "LGBModel":
            return True
        return False

    artifact = ens_workflow(
        RecorderCollector(exp_name=exp_name, rec_key_func=rec_key, rec_filter_func=my_filter),
        RollingGroup(),
    )
    print(artifact)


def main(
    provider_uri="~/.qlib/qlib_data/cn_data",
    task_url="mongodb://10.0.0.4:27017/",
    task_db_name="rolling_db",
    experiment_name="rolling_exp",
    task_pool="rolling_task",
):
    mongo_conf = {
        "task_url": task_url,
        "task_db_name": task_db_name,
    }
    qlib.init(provider_uri=provider_uri, region=REG_CN, mongo=mongo_conf)

    reset(task_pool, experiment_name)
    tasks = task_generating()
    task_training(tasks, task_pool, experiment_name)
    task_collecting(task_pool, experiment_name)


if __name__ == "__main__":
    ## to see the whole process with your own parameters, use the command below
    # python update_online_pred.py main --experiment_name="your_exp_name"
    fire.Fire()

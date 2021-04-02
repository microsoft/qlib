from pprint import pprint

import fire
import qlib
from qlib.config import REG_CN
from qlib.model.trainer import task_train
from qlib.workflow import R
from qlib.workflow.task.collect import RecorderCollector
from qlib.model.ens.ensemble import RollingEnsemble, ens_workflow
from qlib.workflow.task.gen import RollingGen, task_generator
from qlib.workflow.task.manage import TaskManager, run_task
from qlib.workflow.online.manager import RollingOnlineManager
from qlib.workflow.task.utils import list_recorders
from qlib.model.trainer import TrainerRM
from qlib.model.ens.group import RollingGroup

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


def print_online_model():
    print("========== print_online_model ==========")
    print("Current 'online' model:")
    for rid, rec in list_recorders(exp_name).items():
        if rolling_online_manager.get_online_tag(rec) == rolling_online_manager.ONLINE_TAG:
            print(rid)
    print("Current 'next online' model:")
    for rid, rec in list_recorders(exp_name).items():
        if rolling_online_manager.get_online_tag(rec) == rolling_online_manager.NEXT_ONLINE_TAG:
            print(rid)


# This part corresponds to "Task Generating" in the document
def task_generating():

    print("========== task_generating ==========")

    tasks = task_generator(
        tasks=[task_xgboost_config, task_lgb_config],
        generators=rolling_gen,  # generate different date segment
    )

    pprint(tasks)

    return tasks


def task_training(tasks):
    trainer.train(tasks, exp_name, task_pool)


# This part corresponds to "Task Collecting" in the document
def task_collecting():
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
        RecorderCollector(exp_name=exp_name, rec_key_func=rec_key), RollingGroup(), rec_filter_func=my_filter
    )
    print(artifact)


# Reset all things to the first status, be careful to save important data
def reset():
    print("========== reset ==========")
    task_manager.remove()
    exp = R.get_exp(experiment_name=exp_name)
    for rid in exp.list_recorders():
        exp.delete_recorder(rid)


# Run this firstly to see the workflow in Task Management
def first_run():
    print("========== first_run ==========")
    reset()

    tasks = task_generating()
    task_training(tasks)
    task_collecting()

    latest_rec, _ = rolling_online_manager.list_latest_recorders()
    rolling_online_manager.reset_online_tag(latest_rec.values())


def routine():
    print("========== routine ==========")
    print_online_model()
    rolling_online_manager.routine()
    print_online_model()
    task_collecting()


if __name__ == "__main__":
    ####### to train the first version's models, use the command below
    # python task_manager_rolling_with_updating.py first_run

    ####### to update the models and predictions after the trading time, use the command below
    # python task_manager_rolling_with_updating.py after_day

    #################### you need to finish the configurations below #########################

    provider_uri = "~/.qlib/qlib_data/cn_data"  # data_dir
    mongo_conf = {
        "task_url": "mongodb://10.0.0.4:27017/",  # your MongoDB url
        "task_db_name": "rolling_db",  # database name
    }
    qlib.init(provider_uri=provider_uri, region=REG_CN, mongo=mongo_conf)

    exp_name = "rolling_exp"  # experiment name, will be used as the experiment in MLflow
    task_pool = "rolling_task"  # task pool name, will be used as the document in MongoDB
    rolling_step = 550

    ##########################################################################################
    rolling_gen = RollingGen(step=rolling_step, rtype=RollingGen.ROLL_SD)
    task_manager = TaskManager(task_pool=task_pool)
    trainer = TrainerRM()
    rolling_online_manager = RollingOnlineManager(
        experiment_name=exp_name, rolling_gen=rolling_gen, task_manager=task_manager, trainer=trainer
    )

    fire.Fire()

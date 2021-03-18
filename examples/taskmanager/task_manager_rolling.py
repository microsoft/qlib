import qlib
from qlib.config import REG_CN
from qlib.workflow.task.gen import RollingGen, task_generator
from qlib.workflow.task.manage import TaskManager
from qlib.config import C
from qlib.workflow.task.manage import run_task
from qlib.workflow.task.collect import RollingCollector
from qlib.model.trainer import task_train
from qlib.workflow import R
from pprint import pprint

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
def reset():
    print("========== reset ==========")
    TaskManager(task_pool=task_pool).remove()

    # exp = R.get_exp(experiment_name=exp_name)

    # for rid in R.list_recorders():
    #     exp.delete_recorder(rid)


# This part corresponds to "Task Generating" in the document
def task_generating():

    print("========== task_generating ==========")

    tasks = task_generator(
        tasks=[task_xgboost_config, task_lgb_config],
        generators=RollingGen(step=550, rtype=RollingGen.ROLL_SD),  # generate different date segment
    )

    pprint(tasks)

    return tasks


# This part corresponds to "Task Storing" in the document
def task_storing(tasks):
    print("========== task_storing ==========")
    tm = TaskManager(task_pool=task_pool)
    tm.create_task(tasks)  # all tasks will be saved to MongoDB


# This part corresponds to "Task Running" in the document
def task_running():
    print("========== task_running ==========")
    run_task(task_train, task_pool, experiment_name=exp_name)  # all tasks will be trained using "task_train" method


# This part corresponds to "Task Collecting" in the document
def task_collecting():
    print("========== task_collecting ==========")

    def get_task_key(task_config):
        return task_config["model"]["class"]

    def my_filter(recorder):
        # only choose the results of "LGBModel"
        task_key = get_task_key(rolling_collector.get_task(recorder))
        if task_key == "LGBModel":
            return True
        return False

    rolling_collector = RollingCollector(exp_name)
    # group tasks by "get_task_key" and filter tasks by "my_filter"
    pred_rolling = rolling_collector.collect_rolling_predictions(get_task_key, my_filter)
    print(pred_rolling)


if __name__ == "__main__":

    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    mongo_conf = {
        "task_url": "mongodb://10.0.0.4:27017/",  # maybe you need to change it to your url
        "task_db_name": "rolling_db",
    }
    exp_name = "rolling_exp"  # experiment name, will be used as the experiment in MLflow
    task_pool = "rolling_task"  # task pool name, will be used as the document in MongoDB
    qlib.init(provider_uri=provider_uri, region=REG_CN, mongo=mongo_conf)

    reset()
    tasks = task_generating()
    task_storing(tasks)
    task_running()
    task_collecting()

import qlib
from qlib.config import REG_CN
from qlib.workflow.task.gen import RollingGen, task_generator
from qlib.workflow.task.manage import TaskManager
from qlib.config import C

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
        "handler": {"class": "Alpha158", "module_path": "qlib.contrib.data.handler", "kwargs": data_handler_config,},
        "segments": {
            "train": ("2008-01-01", "2014-12-31"),
            "valid": ("2015-01-01", "2016-12-31"),
            "test": ("2017-01-01", "2020-08-01"),
        },
    },
}

record_config = [
    {"class": "SignalRecord", "module_path": "qlib.workflow.record_temp",},
    {"class": "SigAnaRecord", "module_path": "qlib.workflow.record_temp",},
]

# use lgb
task_lgb_config = {
    "model": {"class": "LGBModel", "module_path": "qlib.contrib.model.gbdt",},
    "dataset": dataset_config,
    "record": record_config,
}

# use xgboost
task_xgboost_config = {
    "model": {"class": "XGBModel", "module_path": "qlib.contrib.model.xgboost",},
    "dataset": dataset_config,
    "record": record_config,
}

provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)

C["mongo"] = {
    "task_url": "mongodb://localhost:27017/",  # maybe you need to change it to your url
    "task_db_name": "rolling_db",
}

exp_name = "rolling_exp"  # experiment name, will be used as the experiment in MLflow
task_pool = "rolling_task"  # task pool name, will be used as the document in MongoDB

tasks = task_generator(
    task_xgboost_config,  # default task name
    RollingGen(step=550, rtype=RollingGen.ROLL_SD),  # generate different date segment
    task_lgb=task_lgb_config,  # use "task_lgb" as the task name
)

# Uncomment next two lines to see the generated tasks
# from pprint import pprint
# pprint(tasks)

tm = TaskManager(task_pool=task_pool)
tm.create_task(tasks)  # all tasks will be saved to MongoDB

from qlib.workflow.task.manage import run_task
from qlib.workflow.task.collect import TaskCollector
from qlib.model.trainer import task_train

run_task(task_train, task_pool, experiment_name=exp_name)  # all tasks will be trained using "task_train" method


def get_task_key(task_config):
    task_key = task_config["task_key"]
    rolling_end_timestamp = task_config["dataset"]["kwargs"]["segments"]["test"][1]
    return task_key, rolling_end_timestamp.strftime("%Y-%m-%d")


def my_filter(task_config):
    # only choose the results of "task_lgb" and test in 2019 from all tasks
    task_key, rolling_end = get_task_key(task_config)
    if task_key == "task_lgb" and rolling_end.startswith("2019"):
        return True
    return False


# name tasks by "get_task_key" and filter tasks by "my_filter"
pred_rolling = TaskCollector.collect_predictions(exp_name, get_task_key, my_filter)
pred_rolling

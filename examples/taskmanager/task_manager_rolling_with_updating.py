import qlib
import fire
import mlflow
from qlib.config import C
from qlib.workflow import R
from qlib.config import REG_CN
from qlib.model.trainer import task_train
from qlib.workflow.task.manage import run_task
from qlib.workflow.task.manage import TaskManager
from qlib.workflow.task.utils import TimeAdjuster
from qlib.workflow.task.update import ModelUpdater
from qlib.workflow.task.collect import TaskCollector
from qlib.workflow.task.gen import RollingGen, task_generator


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
            "test": ("2016-01-01", "2017-01-01"),
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

# This part corresponds to "Task Generating" in the document
def task_generating(**kwargs):
    print("========================================= task_generating =========================================")

    rolling_generator = RollingGen(step=rolling_step, rtype=RollingGen.ROLL_EX)

    tasks = task_generator(rolling_generator, **kwargs)

    # See the generated tasks in a easy way
    from pprint import pprint

    pprint(tasks)

    return tasks


# This part corresponds to "Task Storing" in the document
def task_storing(tasks):
    print("========================================= task_storing =========================================")
    tm = TaskManager(task_pool=task_pool)
    tm.create_task(tasks)  # all tasks will be saved to MongoDB


# This part corresponds to "Task Running" in the document
def task_running():
    print("========================================= task_running =========================================")
    run_task(task_train, task_pool, experiment_name=exp_name)  # all tasks will be trained using "task_train" method


# This part corresponds to "Task Collecting" in the document
def task_collecting():
    print("========================================= task_collecting =========================================")

    def get_task_key(task_config):
        task_key = task_config["task_key"]
        rolling_end_timestamp = task_config["dataset"]["kwargs"]["segments"]["test"][1]
        if rolling_end_timestamp == None:
            rolling_end_timestamp = TimeAdjuster().last_date()
        return task_key, rolling_end_timestamp.strftime("%Y-%m-%d")

    def lgb_filter(task_config):
        # only choose the results of "task_lgb"
        task_key, rolling_end = get_task_key(task_config)
        if task_key == "task_lgb":
            return True
        return False

    task_collector = TaskCollector(exp_name)
    pred_rolling = task_collector.collect_predictions(
        get_task_key, lgb_filter
    )  # name tasks by "get_task_key" and filter tasks by "my_filter"
    print(pred_rolling)


# Reset all things to the first status, be careful to save important data
def reset(force_end=False):
    print("========================================= reset =========================================")
    TaskManager(task_pool=task_pool).remove()

    exp = R.get_exp(experiment_name=exp_name)
    recs = TaskCollector(exp_name).list_recorders(only_finished=True)

    for rid in recs:
        exp.delete_recorder(rid)

    try:
        if force_end:
            mlflow.end_run()
    except Exception:
        pass


def set_online_model_to_latest():
    print(
        "========================================= set_online_model_to_latest ========================================="
    )
    model_updater = ModelUpdater(experiment_name=exp_name)
    latest_records, latest_test = model_updater.collect_latest_records()
    model_updater.reset_online_model(latest_records.values())


# Run this firstly to see the workflow in Task Management
def first_run():
    print("========================================= first_run =========================================")
    reset(force_end=True)

    # use "task_lgb" and "task_xgboost" as the task name
    tasks = task_generating(**{"task_xgboost": task_xgboost_config, "task_lgb": task_lgb_config})
    task_storing(tasks)
    task_running()
    task_collecting()
    set_online_model_to_latest()


# Update the predictions of online model
def update_predictions():
    print("========================================= update_predictions =========================================")
    model_updater = ModelUpdater(experiment_name=exp_name)
    model_updater.update_online_pred()


# Update the models using the latest date and set them to online model
def update_model():
    print("========================================= update_model =========================================")
    # get the latest recorders
    model_updater = ModelUpdater(experiment_name=exp_name)
    latest_records, latest_test = model_updater.collect_latest_records()
    # date adjustment based on trade day of Calendar in Qlib
    time_adjuster = TimeAdjuster()
    calendar_latest = time_adjuster.last_date()
    print("The latest date is ", calendar_latest)
    if time_adjuster.cal_interval(calendar_latest, latest_test[0]) > rolling_step:
        print("Need update models!")
        tasks = {}
        for rid, rec in latest_records.items():
            old_task = rec.task
            test_begin = old_task["dataset"]["kwargs"]["segments"]["test"][0]
            # modify the test segment to generate new tasks
            old_task["dataset"]["kwargs"]["segments"]["test"] = (test_begin, calendar_latest)
            tasks[old_task["task_key"]] = old_task

        # retrain the latest model
        new_tasks = task_generating(**tasks)
        task_storing(new_tasks)
        task_running()
        task_collecting()
        latest_records, _ = model_updater.collect_latest_records()

    # set the latest model to online model
    model_updater.reset_online_model(latest_records.values())


# Run whole workflow completely
def whole_workflow():
    print("========================================= whole_workflow =========================================")
    # run this at the first time
    first_run()
    # run this every day
    update_predictions()
    # run this every "rolling_steps" day
    update_model()


if __name__ == "__main__":
    ####### to train the first version's models, use the command below
    # python task_manager_rolling_with_updating.py first_run

    ####### to update the models using the latest date and set them to online model, use the command below
    # python task_manager_rolling_with_updating.py update_model

    ####### to update the predictions to the latest date, use the command below
    # python task_manager_rolling_with_updating.py update_predictions

    ####### to run whole workflow completely, use the command below
    # python task_manager_rolling_with_updating.py whole_workflow

    #################### you need to finish the configurations below #########################

    provider_uri = "~/.qlib/qlib_data/cn_data"  # data_dir
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    C["mongo"] = {
        "task_url": "mongodb://localhost:27017/",  # your MongoDB url
        "task_db_name": "rolling_db",  # database name
    }

    exp_name = "rolling_exp"  # experiment name, will be used as the experiment in MLflow
    task_pool = "rolling_task"  # task pool name, will be used as the document in MongoDB
    rolling_step = 550

    ##########################################################################################

    fire.Fire()

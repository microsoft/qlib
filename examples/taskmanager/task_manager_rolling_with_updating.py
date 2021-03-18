import qlib
import fire
import mlflow
from qlib.config import C
from qlib.workflow import R
from pprint import pprint
from qlib.config import REG_CN
from qlib.model.trainer import task_train
from qlib.workflow.task.manage import run_task
from qlib.workflow.task.manage import TaskManager
from qlib.workflow.task.collect import RollingCollector
from qlib.workflow.task.gen import RollingGen, task_generator
from qlib.workflow.task.online import RollingOnlineManager

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
    print("Current 'online' model:")
    for online in rolling_online_manager.list_online_model().values():
        print(online.info["id"])
    print("Current 'next online' model:")
    for online in rolling_online_manager.list_next_online_model().values():
        print(online.info["id"])


# This part corresponds to "Task Generating" in the document
def task_generating():

    print("========== task_generating ==========")

    tasks = task_generator(
        tasks=[task_xgboost_config, task_lgb_config],
        generators=rolling_gen,  # generate different date segment
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


# Reset all things to the first status, be careful to save important data
def reset(force_end=False):
    print("========== reset ==========")
    task_manager.remove()
    for error in task_manager.query():
        assert False
    exp = R.get_exp(experiment_name=exp_name)
    recs = exp.list_recorders()

    for rid in recs:
        exp.delete_recorder(rid)

    try:
        if force_end:
            mlflow.end_run()
    except Exception:
        pass


# Run this firstly to see the workflow in Task Management
def first_run():
    print("========== first_run ==========")
    reset(force_end=True)

    tasks = task_generating()
    task_storing(tasks)
    task_running()
    task_collecting()

    rolling_online_manager.set_latest_model_to_next_online()
    rolling_online_manager.reset_online_model()


# Update the predictions of online model
def update_predictions():
    print("========== update_predictions ==========")
    rolling_online_manager.update_online_pred()
    task_collecting()
    # if there are some next_online_model, then online them. if no, still use current online_model.
    print_online_model()
    rolling_online_manager.reset_online_model()
    print_online_model()


# Update the models using the latest date and set them to online model
def update_model():
    print("========== update_model ==========")
    rolling_online_manager.prepare_new_models()
    print_online_model()
    rolling_online_manager.set_latest_model_to_next_online()
    print_online_model()


def after_day():
    rolling_online_manager.prepare_signals()
    update_model()
    update_predictions()


# Run whole workflow completely
def whole_workflow():
    print("========== whole_workflow ==========")
    # run this at the first time
    first_run()
    # run this every day after trading
    after_day()


if __name__ == "__main__":
    ####### to train the first version's models, use the command below
    # python task_manager_rolling_with_updating.py first_run

    ####### to update the models using the latest date, use the command below
    # python task_manager_rolling_with_updating.py update_model

    ####### to update the predictions to the latest date, use the command below
    # python task_manager_rolling_with_updating.py update_predictions

    ####### to run whole workflow completely, use the command below
    # python task_manager_rolling_with_updating.py whole_workflow

    #################### you need to finish the configurations below #########################

    provider_uri = "~/.qlib/qlib_data/cn_data"  # data_dir
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    C["mongo"] = {
        "task_url": "mongodb://10.0.0.4:27017/",  # your MongoDB url
        "task_db_name": "online",  # database name
    }

    exp_name = "rolling_exp"  # experiment name, will be used as the experiment in MLflow
    task_pool = "rolling_task"  # task pool name, will be used as the document in MongoDB
    rolling_step = 550

    ##########################################################################################
    rolling_gen = RollingGen(step=550, rtype=RollingGen.ROLL_SD)
    rolling_online_manager = RollingOnlineManager(
        experiment_name=exp_name, rolling_gen=rolling_gen, task_pool=task_pool
    )
    task_manager = TaskManager(task_pool=task_pool)
    fire.Fire()

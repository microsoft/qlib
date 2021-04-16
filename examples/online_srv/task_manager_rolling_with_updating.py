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


class RollingOnlineExample:
    def __init__(
        self,
        exp_name="rolling_exp",
        task_pool="rolling_task",
        provider_uri="~/.qlib/qlib_data/cn_data",
        region="cn",
        task_url="mongodb://10.0.0.4:27017/",
        task_db_name="rolling_db",
        rolling_step=550,
    ):
        self.exp_name = exp_name
        self.task_pool = task_pool
        mongo_conf = {
            "task_url": task_url,  # your MongoDB url
            "task_db_name": task_db_name,  # database name
        }
        qlib.init(provider_uri=provider_uri, region=region, mongo=mongo_conf)

        self.rolling_gen = RollingGen(step=rolling_step, rtype=RollingGen.ROLL_SD)
        self.trainer = TrainerRM(self.exp_name, self.task_pool)
        self.task_manager = TaskManager(self.task_pool)
        self.rolling_online_manager = RollingOnlineManager(
            experiment_name=exp_name, rolling_gen=self.rolling_gen, trainer=self.trainer
        )

    def print_online_model(self):
        print("========== print_online_model ==========")
        print("Current 'online' model:")

        for rec in self.rolling_online_manager.online_models():
            print(rec.info["id"])
        print("Current 'next online' model:")
        for rid, rec in list_recorders(self.exp_name).items():
            if self.rolling_online_manager.get_online_tag(rec) == self.rolling_online_manager.NEXT_ONLINE_TAG:
                print(rid)

    # This part corresponds to "Task Generating" in the document
    def task_generating(self):

        print("========== task_generating ==========")

        tasks = task_generator(
            tasks=[task_xgboost_config, task_lgb_config],
            generators=self.rolling_gen,  # generate different date segment
        )

        pprint(tasks)

        return tasks

    def task_training(self, tasks):
        # self.trainer.train(tasks)
        self.rolling_online_manager.prepare_new_models(tasks, tag=RollingOnlineManager.ONLINE_TAG)

    # This part corresponds to "Task Collecting" in the document
    def task_collecting(self):
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
            RecorderCollector(exp_name=self.exp_name, rec_key_func=rec_key, rec_filter_func=my_filter), RollingGroup()
        )
        print(artifact)

    # Reset all things to the first status, be careful to save important data
    def reset(self):
        print("========== reset ==========")
        self.task_manager.remove()
        exp = R.get_exp(experiment_name=self.exp_name)
        for rid in exp.list_recorders():
            exp.delete_recorder(rid)

    # Run this firstly to see the workflow in Task Management
    def first_run(self):
        print("========== first_run ==========")
        self.reset()

        tasks = self.task_generating()
        pprint(tasks)
        self.task_training(tasks)
        self.task_collecting()

        # latest_rec, _ = self.rolling_online_manager.list_latest_recorders()
        # self.rolling_online_manager.reset_online_tag(list(latest_rec.values()))

    def routine(self):
        print("========== routine ==========")
        self.print_online_model()
        self.rolling_online_manager.routine()
        self.print_online_model()
        self.task_collecting()

    def main(self):
        self.first_run()
        self.routine()


if __name__ == "__main__":
    ####### to train the first version's models, use the command below
    # python task_manager_rolling_with_updating.py first_run

    ####### to update the models and predictions after the trading time, use the command below
    # python task_manager_rolling_with_updating.py after_day

    ####### to define your own parameters, use `--`
    # python task_manager_rolling_with_updating.py first_run --exp_name='your_exp_name' --rolling_step=40
    fire.Fire(RollingOnlineExample)

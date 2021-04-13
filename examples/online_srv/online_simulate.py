from abc import abstractmethod
import copy
from pprint import pprint

import fire
import qlib
from qlib.config import REG_CN
from qlib.model.trainer import task_train
from qlib.workflow import R
from qlib.workflow.task.gen import TaskGen
from qlib.workflow.online.simulator import OnlineSimulator
from qlib.workflow.task.collect import RecorderCollector
from qlib.model.ens.ensemble import RollingEnsemble, ens_workflow
from qlib.workflow.task.gen import RollingGen, task_generator
from qlib.workflow.task.manage import TaskManager, run_task
from qlib.workflow.online.manager import RollingOnlineManager
from qlib.workflow.task.utils import TimeAdjuster, list_recorders
from qlib.model.trainer import TrainerRM
from qlib.model.ens.group import RollingGroup

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


class OnlineSimulatorExample:
    def __init__(
        self,
        exp_name="rolling_exp",
        task_pool="rolling_task",
        provider_uri="~/.qlib/qlib_data/cn_data",
        region="cn",
        task_url="mongodb://10.0.0.4:27017/",
        task_db_name="rolling_db",
        rolling_step=80,
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
            experiment_name=exp_name, rolling_gen=self.rolling_gen, trainer=self.trainer, need_log=False
        )

    # Reset all things to the first status, be careful to save important data
    def reset(self):
        print("========== reset ==========")
        self.task_manager.remove()
        exp = R.get_exp(experiment_name=self.exp_name)
        for rid in exp.list_recorders():
            exp.delete_recorder(rid)

    @staticmethod
    def rec_key(recorder):
        task_config = recorder.load_object("task")
        model_key = task_config["model"]["class"]
        rolling_key = task_config["dataset"]["kwargs"]["segments"]["test"]
        return model_key, rolling_key

    # Run this firstly to see the workflow in Task Management
    def first_run(self):
        print("========== first_run ==========")
        self.reset()

        tasks = task_generator(
            tasks=task_xgboost_config,
            generators=[self.rolling_gen],  # generate different date segment
        )

        pprint(tasks)

        self.trainer.train(tasks)

        print("========== task collecting ==========")

        artifact = ens_workflow(RecorderCollector(exp_name=self.exp_name, rec_key_func=self.rec_key), RollingGroup())
        print(artifact)

        latest_rec, _ = self.rolling_online_manager.list_latest_recorders()
        self.rolling_online_manager.set_online_tag(RollingOnlineManager.ONLINE_TAG, list(latest_rec.values()))

    def simulate(self):

        print("========== simulate ==========")
        onlinesimulator = OnlineSimulator(
            start_time="2018-09-10",
            end_time="2018-10-31",
            onlinemanager=self.rolling_online_manager,
            collector=RecorderCollector(exp_name=self.exp_name, rec_key_func=self.rec_key),
            process_list=[RollingGroup()],
        )
        results = onlinesimulator.simulate()
        print(results)
        recs_dict = onlinesimulator.online_models()
        for time, recs in recs_dict.items():
            print(f"{str(time[0])} to {str(time[1])}:")
            for rec in recs:
                print(rec.info["id"])


if __name__ == "__main__":
    ose = OnlineSimulatorExample()
    ose.first_run()
    ose.simulate()

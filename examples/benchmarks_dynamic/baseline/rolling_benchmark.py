# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from qlib.model.ens.ensemble import RollingEnsemble
from qlib.utils import init_instance_by_config
import fire
import yaml
from qlib import auto_init
from pathlib import Path
from tqdm.auto import tqdm
from qlib.model.trainer import TrainerR
from qlib.workflow import R
from qlib.tests.data import GetData

DIRNAME = Path(__file__).absolute().resolve().parent
from qlib.workflow.task.gen import task_generator, RollingGen
from qlib.workflow.task.collect import RecorderCollector
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord


class RollingBenchmark:
    """
    **NOTE**
    before running the example, please clean your previous results with following command
    - `rm -r mlruns`

    """

    def __init__(self, rolling_exp="rolling_models", model_type="linear") -> None:
        self.step = 20
        self.horizon = 20
        self.rolling_exp = rolling_exp
        self.model_type = model_type

    def basic_task(self):
        """For fast training rolling"""
        if self.model_type == "gbdt":
            conf_path = DIRNAME.parent.parent / "benchmarks" / "LightGBM" / "workflow_config_lightgbm_Alpha158.yaml"
            # dump the processed data on to disk for later loading to speed up the processing
            h_path = DIRNAME / "lightgbm_alpha158_handler_horizon{}.pkl".format(self.horizon)
        elif self.model_type == "linear":
            conf_path = DIRNAME.parent.parent / "benchmarks" / "Linear" / "workflow_config_linear_Alpha158.yaml"
            h_path = DIRNAME / "linear_alpha158_handler_horizon{}.pkl".format(self.horizon)
        else:
            raise AssertionError("Model type is not supported!")
        with conf_path.open("r") as f:
            conf = yaml.safe_load(f)

        # modify dataset horizon
        conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["label"] = [
            "Ref($close, -{}) / Ref($close, -1) - 1".format(self.horizon + 1)
        ]

        task = conf["task"]

        if not h_path.exists():
            h_conf = task["dataset"]["kwargs"]["handler"]
            h = init_instance_by_config(h_conf)
            h.to_pickle(h_path, dump_all=True)

        task["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
        task["record"] = ["qlib.workflow.record_temp.SignalRecord"]
        return task

    def create_rolling_tasks(self):
        task = self.basic_task()
        task_l = task_generator(
            task, RollingGen(step=self.step, trunc_days=self.horizon + 1)
        )  # the last two days should be truncated to avoid information leakage
        return task_l

    def train_rolling_tasks(self, task_l=None):
        if task_l is None:
            task_l = self.create_rolling_tasks()
        trainer = TrainerR(experiment_name=self.rolling_exp)
        trainer(task_l)

    COMB_EXP = "rolling"

    def ens_rolling(self):
        rc = RecorderCollector(
            experiment=self.rolling_exp,
            artifacts_key=["pred", "label"],
            process_list=[RollingEnsemble()],
            # rec_key_func=lambda rec: (self.COMB_EXP, rec.info["id"]),
            artifacts_path={"pred": "pred.pkl", "label": "label.pkl"},
        )
        res = rc()
        with R.start(experiment_name=self.COMB_EXP):
            R.log_params(exp_name=self.rolling_exp)
            R.save_objects(**{"pred.pkl": res["pred"], "label.pkl": res["label"]})

    def update_rolling_rec(self):
        """
        Evaluate the combined rolling results
        """
        for rid, rec in R.list_recorders(experiment_name=self.COMB_EXP).items():
            for rt_cls in SigAnaRecord, PortAnaRecord:
                rt = rt_cls(recorder=rec, skip_existing=True)
                rt.generate()
        print(f"Your evaluation results can be found in the experiment named `{self.COMB_EXP}`.")

    def run_all(self):
        # the results will be  save in mlruns.
        # 1) each rolling task is saved in rolling_models
        self.train_rolling_tasks()
        # 2) combined rolling tasks and evaluation results are saved in rolling
        self.ens_rolling()
        self.update_rolling_rec()


if __name__ == "__main__":
    GetData().qlib_data(exists_skip=True)
    auto_init()
    fire.Fire(RollingBenchmark)

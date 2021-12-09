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

DIRNAME = Path(__file__).absolute().resolve().parent
from qlib.workflow.task.gen import task_generator, RollingGen
from qlib.workflow.task.collect import RecorderCollector


class RollingBenchmark:
    """
    before running the example, please clean your previous results with following command
    - `rm -r mlruns`

    """
    def __init__(self) -> None:
        self.step = 20

    def basic_task(self):
        """For fast training rolling"""
        conf_path = DIRNAME.parent.parent / "benchmarks" / "LightGBM" / "workflow_config_lightgbm_Alpha158.yaml"
        with conf_path.open("r") as f:
            conf = yaml.safe_load(f)
        task = conf["task"]

        # dump the processed data on to disk for later loading to speed up the processing
        h_path = DIRNAME / "lightgbm_alpha158_handler.pkl"

        if not h_path.exists():
            h_conf = task["dataset"]["kwargs"]["handler"]
            h = init_instance_by_config(h_conf)
            h.to_pickle(h_path, dump_all=True)

        task["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
        task["record"] = ["qlib.workflow.record_temp.SignalRecord"]
        return task

    def create_rolling_tasks(self):
        task = self.basic_task()
        task_l = task_generator(task, RollingGen(
            step=self.step, trunc_days=2))  # the last two days should be truncated to avoid information leakage
        return task_l

    def run_rolling_tasks(self):
        task_l = self.create_rolling_tasks()
        trainer = TrainerR(experiment_name="rolling_models")
        trainer(task_l)

    def ens_rolling(self):
        comb_key = "rolling"
        rc = RecorderCollector(experiment="rolling_models",
                               artifacts_key=["pred", "label"],
                               process_list=[RollingEnsemble()],
                               # rec_key_func=lambda rec: (comb_key, rec.info["id"]),
                               artifacts_path={
                                   "pred": "pred.pkl",
                                   "label": "label.pkl"
                               })
        res = rc()
        with R.start(experiment_name=comb_key):
            R.save_objects(**{"pred.pkl": res["pred"], "label.pkl": res["label"]})

    def update_rolling_rec(self):
        pass


if __name__ == "__main__":
    auto_init()
    fire.Fire(RollingBenchmark)

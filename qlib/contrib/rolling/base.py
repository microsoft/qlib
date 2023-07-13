# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Optional, List, Union
from qlib.model.ens.ensemble import RollingEnsemble
from qlib.utils import init_instance_by_config, get_cls_kwargs
import fire
import yaml
import pandas as pd
from qlib import auto_init
from pathlib import Path
from qlib.model.trainer import TrainerR
from qlib.log import get_module_logger
from qlib.utils.data import update_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord

from qlib.workflow.task.gen import task_generator, RollingGen
from qlib.workflow.task.collect import RecorderCollector


class Rolling:
    """
    The motivation of Rolling Module
    - It only focus **offlinely** turn a specific task to rollinng
    - To make the implementation easier, following factors are ignored.
        - The tasks is dependent (e.g. time series).

    Related modules and difference from me:
    - MetaController: It is learning how to handle a task (e.g. learning to learn).
        - But rolling is about how to split a single task into tasks in time series and run them.
    - OnlineStrategy: It is focusing on serving a model, the model can be updated time dependently in time.
        - Rolling is much simpler and is only for testing rolling models offline. It does not want to share the interface with OnlineStrategy.

    The code about rolling is shared in `task_generator` & `RollingGen` level between me and the above modules
    But it is for different purpose, so other parts are not shared.


    .. code-block:: shell

        # here is an typical use case of the module.
        python -m qlib.contrib.rolling.base --conf_path <path to the yaml> run

    **NOTE**
    before running the example, please clean your previous results with following command
    - `rm -r mlruns`
    - Because it is very hard to permanently delete a experiment (it will be moved into .trash and raise error when creating experiment with same name).

    """

    def __init__(
        self,
        conf_path: Union[str, Path],
        exp_name: Optional[str] = None,
        horizon: Optional[int] = 20,
        step: int = 20,
        h_path: Optional[str] = None,
        train_start: Optional[str] = None,
        test_end: Optional[str] = None,
        task_ext_conf: Optional[dict] = None,
        enable_handler_cache: bool = True,
        rolling_exp: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        conf_path : str
            Path to the config for rolling.
        exp_name : Optional[str]
            The exp name of the outputs (Output is a record which contains the concatenated predictions of rolling records).
        horizon: Optional[int] = 20,
            The horizon of the prediction target.
            This is used to override the prediction horizon of the file.
        h_path : Optional[str]
            the dumped data handler;
            It may come from other data source. It will override the data handler in the config.
        test_end : Optional[str]
            the test end for the data. It is typically used together with the handler
            You can do the same thing with task_ext_conf in a more complicated way
        train_start : Optional[str]
            the train start for the data.  It is typically used together with the handler.
            You can do the same thing with task_ext_conf in a more complicated way
        task_ext_conf : Optional[dict]
            some option to update the task config.
        rolling_exp : Optional[str]
            The name for the experiments for rolling.
            It will contains a lot of record in an experiment. Each record corresponds to a specific rolling.
            Please note that it is different from the final experiments
        """
        self.conf_path = Path(conf_path)
        self.exp_name = exp_name
        self._rid = None  # the final combined recorder id in `exp_name`

        self.step = step
        assert horizon is not None, "Current version does not support extracting horizon from the underlying dataset"
        self.horizon = horizon
        if rolling_exp is None:
            datetime_suffix = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
            self.rolling_exp = f"rolling_models_{datetime_suffix}"
        else:
            self.rolling_exp = rolling_exp
            self.logger.warning(
                "Using user specifiied name for rolling models. So the experiment names duplicateds. "
                "Please manually remove your experiment for rolling model with command like `rm -r mlruns`."
                " Otherwise it will prevents the creating of experimen with same name"
            )
        self.train_start = train_start
        self.test_end = test_end
        self.logger = get_module_logger("Rolling")
        self.task_ext_conf = task_ext_conf

        self.enable_handler_cache = enable_handler_cache
        if not enable_handler_cache and h_path is not None:
            raise ValueError(
                "If you want to existing hanlder cache(e.g. `h_path`), you must enable `enable_handler_cache`"
            )
        self.h_path = h_path

        # FIXME:
        # - the qlib_init section will be ignored by me.
        # - So we have to design a priority mechanism to solve this issue.

    def _raw_conf(self) -> dict:
        with self.conf_path.open("r") as f:
            return yaml.safe_load(f)

    def _replace_hanler_with_cache(self, task: dict):
        """
        Due to the data processing part in original rolling is slow. So we have to
        This class tries to add more feature
        """
        if self.enable_handler_cache:
            if self.h_path is not None:
                h_path = Path(self.h_path)
            else:
                h_path = self.conf_path.with_suffix(".handler.pkl")
            if not h_path.exists():
                h_conf = task["dataset"]["kwargs"]["handler"]
                h = init_instance_by_config(h_conf)
                h.to_pickle(h_path, dump_all=True)
            task["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
        return task

    def _update_start_end_time(self, task: dict):
        if self.train_start is not None:
            seg = task["dataset"]["kwargs"]["segments"]["train"]
            task["dataset"]["kwargs"]["segments"]["train"] = pd.Timestamp(self.train_start), seg[1]

        if self.test_end is not None:
            seg = task["dataset"]["kwargs"]["segments"]["test"]
            task["dataset"]["kwargs"]["segments"]["test"] = seg[0], pd.Timestamp(self.test_end)
        return task

    def basic_task(self):
        """
        The basic task may not be the exactly same as the config from `conf_path` from __init__ due to
        - some parameters could be overriding by some parameters from __init__
        - user could implementing sublcass to change it for higher performance

        """
        conf = self._raw_conf()

        # modify dataset horizon
        # NOTE:
        # It assumpts that the label can be modifiled in the handler's kwargs
        # But is not always a valid. It is only valid in the predefined dataset `Alpha158` & `Alpha360`
        if self.horizon is None:
            # TODO:
            # - get horizon automatically from the expression!!!!
            raise NotImplementedError(f"This type of input is not supported")
        else:
            self.logger.info("The prediction horizon is overrided")
            conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["label"] = [
                "Ref($close, -{}) / Ref($close, -1) - 1".format(self.horizon + 1)
            ]

        task = conf["task"]
        task = self._replace_hanler_with_cache(task)
        task = self._update_start_end_time(task)

        if self.task_ext_conf is not None:
            task = update_config(task, self.task_ext_conf)
        self.logger.info(task)
        return task

    def get_task_list(self) -> List[dict]:
        """return a batch of tasks for rolling."""
        task = self.basic_task()
        task_l = task_generator(
            task, RollingGen(step=self.step, trunc_days=self.horizon + 1)
        )  # the last two days should be truncated to avoid information leakage
        for t in task_l:
            # when we rolling tasks. No further analyis is needed.
            # analyis are postponed to the final ensemble.
            t["record"] = ["qlib.workflow.record_temp.SignalRecord"]
        return task_l

    def _train_rolling_tasks(self):
        task_l = self.get_task_list()
        self.logger.info("Deleting previous Rolling results")
        try:
            # TODO: mlflow does not support permanently delete experiment
            # it will  be moved to .trash and prevents creating the experiments with the same name
            R.delete_exp(experiment_name=self.rolling_exp)  # We should remove the rolling experiments.
        except ValueError:
            self.logger.info("No previous rolling results")
        trainer = TrainerR(experiment_name=self.rolling_exp)
        trainer(task_l)

    def _ens_rolling(self):
        rc = RecorderCollector(
            experiment=self.rolling_exp,
            artifacts_key=["pred", "label"],
            process_list=[RollingEnsemble()],
            # rec_key_func=lambda rec: (self.COMB_EXP, rec.info["id"]),
            artifacts_path={"pred": "pred.pkl", "label": "label.pkl"},
        )
        res = rc()
        with R.start(experiment_name=self.exp_name):
            R.log_params(exp_name=self.rolling_exp)
            R.save_objects(**{"pred.pkl": res["pred"], "label.pkl": res["label"]})
            self._rid = R.get_recorder().id

    def _update_rolling_rec(self):
        """
        Evaluate the combined rolling results
        """
        rec = R.get_recorder(experiment_name=self.exp_name, recorder_id=self._rid)
        # Follow the original analyser
        records = self._raw_conf()["task"].get("record", [])
        if isinstance(records, dict):  # prevent only one dict
            records = [records]
        for record in records:
            if issubclass(get_cls_kwargs(record)[0], SignalRecord):
                # skip the signal record.
                continue
            r = init_instance_by_config(
                record,
                recorder=rec,
                default_module="qlib.workflow.record_temp",
            )
            r.generate()
        print(f"Your evaluation results can be found in the experiment named `{self.exp_name}`.")

    def run(self):
        # the results will be  save in mlruns.
        # 1) each rolling task is saved in rolling_models
        self._train_rolling_tasks()
        # 2) combined rolling tasks and evaluation results are saved in rolling
        self._ens_rolling()
        self._update_rolling_rec()


if __name__ == "__main__":
    auto_init()
    fire.Fire(Rolling)

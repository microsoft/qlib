# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path
import pickle
from typing import Optional, Union

import pandas as pd
import yaml

from qlib.contrib.meta.data_selection.dataset import InternalData, MetaDatasetDS
from qlib.contrib.meta.data_selection.model import MetaModelDS
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.meta.task import MetaTask
from qlib.model.trainer import TrainerR
from qlib.typehint import Literal
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.task.utils import replace_task_handler_with_cache

from .base import Rolling

# LGBM is designed for feature importance & similarity
LGBM_MODEL = """
class: LGBModel
module_path: qlib.contrib.model.gbdt
kwargs:
    loss: mse
    colsample_bytree: 0.8879
    learning_rate: 0.2
    subsample: 0.8789
    lambda_l1: 205.6999
    lambda_l2: 580.9768
    max_depth: 8
    num_leaves: 210
    num_threads: 20
"""
# covnert the yaml to dict
LGBM_MODEL = yaml.load(LGBM_MODEL, Loader=yaml.FullLoader)

LINEAR_MODEL = """
class: LinearModel
module_path: qlib.contrib.model.linear
kwargs:
    estimator: ridge
    alpha: 0.05
"""
LINEAR_MODEL = yaml.load(LINEAR_MODEL, Loader=yaml.FullLoader)

PROC_ARGS = """
infer_processors:
    - class: RobustZScoreNorm
      kwargs:
          fields_group: feature
          clip_outlier: true
    - class: Fillna
      kwargs:
          fields_group: feature
learn_processors:
    - class: DropnaLabel
    - class: CSRankNorm
      kwargs:
          fields_group: label
"""
PROC_ARGS = yaml.load(PROC_ARGS, Loader=yaml.FullLoader)

UTIL_MODEL_TYPE = Literal["linear", "gbdt"]


class DDGDA(Rolling):
    """
    It is a rolling based on DDG-DA

    **NOTE**
    before running the example, please clean your previous results with following command
    - `rm -r mlruns`
    """

    def __init__(
        self,
        sim_task_model: UTIL_MODEL_TYPE = "gbdt",
        meta_1st_train_end: Optional[str] = None,
        alpha: float = 0.01,
        working_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        sim_task_model: Literal["linear", "gbdt"] = "gbdt",
            The model for calculating similarity between data.
        meta_1st_train_end: Optional[str]
            the datetime of training end of the first meta_task
        alpha: float
            Setting the L2 regularization for ridge
            The `alpha` is only passed to MetaModelDS (it is not passed to sim_task_model currently..)
        """
        # NOTE:
        # the horizon must match the meaning in the base task template
        self.meta_exp_name = "DDG-DA"
        self.sim_task_model: UTIL_MODEL_TYPE = sim_task_model  # The model to capture the distribution of data.
        self.alpha = alpha
        self.meta_1st_train_end = meta_1st_train_end
        super().__init__(**kwargs)
        self.working_dir = self.conf_path.parent if working_dir is None else Path(working_dir)
        self.proxy_hd = self.working_dir / "handler_proxy.pkl"

    def _adjust_task(self, task: dict, astype: UTIL_MODEL_TYPE):
        """
        some task are use for special purpose.
        For example:
        - GBDT for calculating feature importance
        - Linear or GBDT for calculating similarity
        - Datset (well processed) that aligned to Linear that for meta learning
        """
        # NOTE: here is just for aligning with previous implementation
        # It is not necessary for the current implementation
        handler = task["dataset"].setdefault("kwargs", {}).setdefault("handler", {})
        if astype == "gbdt":
            task["model"] = LGBM_MODEL
            if isinstance(handler, dict):
                for k in ["infer_processors", "learn_processors"]:
                    if k in handler.setdefault("kwargs", {}):
                        handler["kwargs"].pop(k)
        elif astype == "linear":
            task["model"] = LINEAR_MODEL
            handler["kwargs"].update(PROC_ARGS)
        else:
            raise ValueError(f"astype not supported: {astype}")
        return task

    def _get_feature_importance(self):
        # this must be lightGBM, because it needs to get the feature importance
        task = self.basic_task(enable_handler_cache=False)
        task = self._adjust_task(task, astype="gbdt")
        task = replace_task_handler_with_cache(task, self.working_dir)

        with R.start(experiment_name="feature_importance"):
            model = init_instance_by_config(task["model"])
            dataset = init_instance_by_config(task["dataset"])
            model.fit(dataset)

        fi = model.get_feature_importance()
        # Because the model use numpy instead of dataframe for training lightgbm
        # So the we must use following extra steps to get the right feature importance
        df = dataset.prepare(segments=slice(None), col_set="feature", data_key=DataHandlerLP.DK_R)
        cols = df.columns
        fi_named = {cols[int(k.split("_")[1])]: imp for k, imp in fi.to_dict().items()}

        return pd.Series(fi_named)

    def _dump_data_for_proxy_model(self):
        """
        Dump data for training meta model.
        The meta model will be trained upon the proxy forecasting model.
        This dataset is for the proxy forecasting model.
        """
        topk = 30
        fi = self._get_feature_importance()
        col_selected = fi.nlargest(topk)
        # NOTE: adjusting to `self.sim_task_model` just for aligning with previous implementation.
        task = self._adjust_task(self.basic_task(enable_handler_cache=False), self.sim_task_model)
        task = replace_task_handler_with_cache(task, self.working_dir)

        dataset = init_instance_by_config(task["dataset"])
        prep_ds = dataset.prepare(slice(None), col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)

        feature_df = prep_ds["feature"]
        label_df = prep_ds["label"]

        feature_selected = feature_df.loc[:, col_selected.index]

        feature_selected = feature_selected.groupby("datetime", group_keys=False).apply(
            lambda df: (df - df.mean()).div(df.std())
        )
        feature_selected = feature_selected.fillna(0.0)

        df_all = {
            "label": label_df.reindex(feature_selected.index),
            "feature": feature_selected,
        }
        df_all = pd.concat(df_all, axis=1)
        df_all.to_pickle(self.working_dir / "fea_label_df.pkl")

        # dump data in handler format for aligning the interface
        handler = DataHandlerLP(
            data_loader={
                "class": "qlib.data.dataset.loader.StaticDataLoader",
                "kwargs": {"config": self.working_dir / "fea_label_df.pkl"},
            }
        )
        handler.to_pickle(self.working_dir / self.proxy_hd, dump_all=True)

    @property
    def _internal_data_path(self):
        return self.working_dir / f"internal_data_s{self.step}.pkl"

    def _dump_meta_ipt(self):
        """
        Dump data for training meta model.
        This function will dump the input data for meta model
        """
        # According to the experiments, the choice of the model type is very important for achieving good results
        sim_task = self._adjust_task(self.basic_task(enable_handler_cache=False), astype=self.sim_task_model)
        sim_task = replace_task_handler_with_cache(sim_task, self.working_dir)

        if self.sim_task_model == "gbdt":
            sim_task["model"].setdefault("kwargs", {}).update({"early_stopping_rounds": None, "num_boost_round": 150})

        exp_name_sim = f"data_sim_s{self.step}"

        internal_data = InternalData(sim_task, self.step, exp_name=exp_name_sim)
        internal_data.setup(trainer=TrainerR)

        with self._internal_data_path.open("wb") as f:
            pickle.dump(internal_data, f)

    def _train_meta_model(self, fill_method="max"):
        """
        training a meta model based on a simplified linear proxy model;
        """

        # 1) leverage the simplified proxy forecasting model to train meta model.
        # - Only the dataset part is important, in current version of meta model will integrate the

        # the train_start for training meta model does not necessarily align with final rolling
        train_start = "2008-01-01" if self.train_start is None else self.train_start
        train_end = "2010-12-31" if self.meta_1st_train_end is None else self.meta_1st_train_end
        test_start = (pd.Timestamp(train_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        proxy_forecast_model_task = {
            # "model": "qlib.contrib.model.linear.LinearModel",
            "dataset": {
                "class": "qlib.data.dataset.DatasetH",
                "kwargs": {
                    "handler": f"file://{(self.working_dir / self.proxy_hd).absolute()}",
                    "segments": {
                        "train": (train_start, train_end),
                        "test": (test_start, self.basic_task()["dataset"]["kwargs"]["segments"]["test"][1]),
                    },
                },
            },
            # "record": ["qlib.workflow.record_temp.SignalRecord"]
        }
        # the proxy_forecast_model_task will be used to create meta tasks.
        # The test date of first task will be 2011-01-01. Each test segment will be about 20days
        # The tasks include all training tasks and test tasks.

        # 2) preparing meta dataset
        kwargs = dict(
            task_tpl=proxy_forecast_model_task,
            step=self.step,
            segments=0.62,  # keep test period consistent with the dataset yaml
            trunc_days=1 + self.horizon,
            hist_step_n=30,
            fill_method=fill_method,
            rolling_ext_days=0,
        )
        # NOTE:
        # the input of meta model (internal data) are shared between proxy model and final forecasting model
        # but their task test segment are not aligned! It worked in my previous experiment.
        # So the misalignment will not affect the effectiveness of the method.
        with self._internal_data_path.open("rb") as f:
            internal_data = pickle.load(f)

        md = MetaDatasetDS(exp_name=internal_data, **kwargs)

        # 3) train and logging meta model
        with R.start(experiment_name=self.meta_exp_name):
            R.log_params(**kwargs)
            mm = MetaModelDS(
                step=self.step, hist_step_n=kwargs["hist_step_n"], lr=0.001, max_epoch=30, seed=43, alpha=self.alpha
            )
            mm.fit(md)
            R.save_objects(model=mm)

    @property
    def _task_path(self):
        return self.working_dir / f"tasks_s{self.step}.pkl"

    def get_task_list(self):
        """
        Leverage meta-model for inference:
        - Given
            - baseline tasks
            - input for meta model(internal data)
            - meta model (its learnt knowledge on proxy forecasting model is expected to transfer to normal forecasting model)
        """
        # 1) get meta model
        exp = R.get_exp(experiment_name=self.meta_exp_name)
        rec = exp.list_recorders(rtype=exp.RT_L)[0]
        meta_model: MetaModelDS = rec.load_object("model")

        # 2)
        # we are transfer to knowledge of meta model to final forecasting tasks.
        # Create MetaTaskDataset for the final forecasting tasks
        # Aligning the setting of it to the MetaTaskDataset when training Meta model is necessary

        # 2.1) get previous config
        param = rec.list_params()
        trunc_days = int(param["trunc_days"])
        step = int(param["step"])
        hist_step_n = int(param["hist_step_n"])
        fill_method = param.get("fill_method", "max")

        task_l = super().get_task_list()

        # 2.2) create meta dataset for final dataset
        kwargs = dict(
            task_tpl=task_l,
            step=step,
            segments=0.0,  # all the tasks are for testing
            trunc_days=trunc_days,
            hist_step_n=hist_step_n,
            fill_method=fill_method,
            task_mode=MetaTask.PROC_MODE_TRANSFER,
        )

        with self._internal_data_path.open("rb") as f:
            internal_data = pickle.load(f)
        mds = MetaDatasetDS(exp_name=internal_data, **kwargs)

        # 3) meta model make inference and get new qlib task
        new_tasks = meta_model.inference(mds)
        with self._task_path.open("wb") as f:
            pickle.dump(new_tasks, f)
        return new_tasks

    def run(self):
        # prepare the meta model for rolling ---------
        # 1) file: handler_proxy.pkl (self.proxy_hd)
        self._dump_data_for_proxy_model()
        # 2)
        # file: internal_data_s20.pkl
        # mlflow: data_sim_s20, models for calculating meta_ipt
        self._dump_meta_ipt()
        # 3) meta model will be stored in `DDG-DA`
        self._train_meta_model()

        # Run rolling --------------------------------
        # 4) new_tasks are saved in "tasks_s20.pkl" (reweighter is added)
        # - the meta inference are done when calling `get_task_list`
        # 5) load the saved tasks and train model
        super().run()

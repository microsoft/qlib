# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
from qlib.model.meta.task import MetaTask
import torch
from torch import nn
from torch import optim
from tqdm.auto import tqdm
import collections
import copy
from typing import Union, List, Tuple, Dict

from ....data.dataset.weight import SampleReweighter, Reweighter
from ....model.meta.dataset import MetaDataset
from ....model.meta.model import MetaModel, MetaTaskModel
from ....workflow import R

from .utils import fill_diagnal, ICLoss
from .dataset import MetaDatasetHDS
from qlib.contrib.meta.data_selection.net import PredNet
from qlib.data.dataset.weight import Reweighter


class TimeReweighter(Reweighter):

    def __init__(self, time_weight: pd.Series):
        self.time_weight = time_weight

    def reweight(self, data: Union[pd.DataFrame, pd.Series]):
        w_s = pd.Series(1., index=data.index)
        for k, w in self.time_weight.items():
            w_s.loc[slice(*k)] = w
        return w_s


class MetaModelDS(MetaTaskModel):
    """
    The meta-model for meta-learning-based data selection.
    """

    def __init__(
        self,
        step,
        hist_step_n,
        clip_method="tanh",
        clip_weight=2.0,
        criterion="ic_loss",
        lr=0.0001,
        max_epoch=150,
    ):
        self.step = step
        self.hist_step_n = hist_step_n
        self.clip_method = clip_method
        self.clip_weight = clip_weight
        self.criterion = criterion
        self.lr = lr
        self.max_epoch = max_epoch
        self.fitted = False

    def run_epoch(self, phase, task_list, epoch, opt, loss_l, ignore_weight=False):
        if phase == "train":  # phase 0 for training, 1 for inference
            self.tn.train()
            torch.set_grad_enabled(True)
        else:
            self.tn.eval()
            torch.set_grad_enabled(False)
        running_loss = 0.0
        pred_y_all = []
        for task in tqdm(task_list, desc=f"{phase} Task", leave=False):
            meta_input = task.get_meta_input()
            pred, weights = self.tn(
                meta_input["X"],
                meta_input["y"],
                meta_input["time_perf"],
                meta_input["time_belong"],
                meta_input["X_test"],
                ignore_weight=ignore_weight
            ) # 这里可能因为如下原因导致pred为None;
            if self.criterion == "mse":
                criterion = nn.MSELoss()
                loss = criterion(pred, meta_input["y_test"])
            elif self.criterion == "ic_loss":
                criterion = ICLoss()
                loss = criterion(pred, meta_input["y_test"], meta_input["test_idx"], skip_size=50)

            if np.isnan(loss.detach().item()): __import__('ipdb').set_trace()

            if phase == "train":
                opt.zero_grad()
                norm_loss = nn.MSELoss()
                loss.backward()
                opt.step()
            elif phase == "test":
                pass

            pred_y_all.append(
                pd.DataFrame(
                    {
                        "pred": pd.Series(pred.detach().cpu().numpy(), index=meta_input["test_idx"]),
                        "label": pd.Series(
                            meta_input["y_test"].detach().cpu().numpy(), index=meta_input["test_idx"]
                        ),
                    }
                )
            )
            running_loss += loss.detach().item()
        running_loss = running_loss / len(task_list)
        loss_l.setdefault(phase, []).append(running_loss)

        pred_y_all = pd.concat(pred_y_all)
        ic = (
            pred_y_all.groupby("datetime")
            .apply(lambda df: df["pred"].corr(df["label"], method="spearman"))
            .mean()
        )

        R.log_metrics(**{f"loss/{phase}": running_loss, "step": epoch})
        R.log_metrics(**{f"ic/{phase}": ic, "step": epoch})

    def fit(self, meta_dataset: MetaDatasetHDS):
        """
        The meta-learning-based data selection interacts directly with meta-dataset due to the close-form proxy measurement.

        Parameters
        ----------
        meta_dataset : MetaDatasetHDS
            The meta-model takes the meta-dataset for its training process.
        """

        if not self.fitted:
            for k in set(["lr", "step", "hist_step_n", "clip_method", "clip_weight", "criterion", "max_epoch"]):
                R.log_params(**{k: getattr(self, k)})

        # FIXME: get test tasks for just checking the performance
        phases = ["train", "test"]
        meta_tasks_l = meta_dataset.prepare_tasks(phases)

        self.tn = PredNet(
            step=self.step, hist_step_n=self.hist_step_n, clip_weight=self.clip_weight, clip_method=self.clip_method
        )

        opt = optim.Adam(self.tn.parameters(), lr=self.lr)

        # run weight with no weight
        for phase, task_list in zip(phases, meta_tasks_l):
            self.run_epoch(f"{phase}_noweight", task_list, 0, opt, {}, ignore_weight=True)
            self.run_epoch(f"{phase}_init", task_list, 0, opt, {})

        # run training
        loss_l = {}
        for epoch in tqdm(range(self.max_epoch), desc="epoch"):
            for phase, task_list in zip(phases, meta_tasks_l):
                self.run_epoch(phase, task_list, epoch, opt, loss_l)
            R.save_objects(**{"model.pkl": self.tn})
        self.fitted = True

    # TODO: refactor
    def _inference_single_task(self, meta_id: tuple, meta_dataset: MetaDatasetHDS):
        meta_task = meta_dataset.get_meta_task_by_test_period(meta_id)
        if meta_task is not None:
            self.tn.eval()
            torch.set_grad_enabled(False)
            (
                X,
                y,
                time_perf,
                time_belong,
                X_test,
                y_test,
                test_idx,
                train_idx,
                test_period,
            ) = meta_task.prepare_task_data()
            weights = self.tn.get_sample_weights(X, time_perf, time_belong)
            reweighter = SampleReweighter(pd.Series(weights.detach().cpu().numpy(), index=train_idx))
            return reweighter
        else:
            raise ValueError("The current task is not supported!")

    # TODO: refactor
    def inference(self, meta_ids: Union[List[tuple], tuple], meta_dataset: MetaDatasetHDS):
        """
        Inference a single task with meta-dataset. The meta-model must be fitted.

        Parameters
        ----------
        tasks: Union[List[dict], dict]
            A list of definitions.
        meta_dataset: MetaDatasetHDS
        """
        if not self.fitted:
            raise ValueError("The meta-model is not fitted yet!")
        if isinstance(meta_ids, tuple):
            return {meta_ids: self._inference_single_task(meta_ids, meta_dataset)}

        elif isinstance(meta_ids, list):
            reweighters = {}
            for meta_id in meta_ids:
                reweighters[meta_id] = self._inference_single_task(meta_id, meta_dataset)
            return reweighters
        else:
            raise NotImplementedError("This type of task definition is not supported!")

    # TODO: refactor
    def prepare_tasks(self, task: Union[List[dict], dict], reweighters: dict):
        """

        Parameters
        ----------
        tasks: Union[List[dict], dict]
            A list of definitions.
        """
        if not self.fitted:
            raise ValueError("The meta-model is not fitted yet!")
        if isinstance(task, dict):
            task_c = copy.deepcopy(task)
            test_period = task_c["dataset"]["kwargs"]["segments"]["test"]
            if test_period in reweighters:
                task_c["reweighter"] = reweighters[test_period]
            else:
                nearest_key = None
                for key in reweighters:
                    if key[0] <= test_period[0]:
                        if nearest_key is None or nearest_key[0] < key[0]:
                            nearest_key = key
                if nearest_key is not None:
                    task_c["reweighter"] = reweighters[nearest_key]
                else:
                    print(
                        "Warning: The task with test period:",
                        test_period,
                        " does not have the corresponding reweighter!",
                    )
            return task_c
        elif isinstance(task, list):
            return [self.prepare_tasks(i, reweighters) for i in task]
        else:
            raise NotImplementedError("This type of task definition is not supported!")

    def prepare_task(self, task: MetaTask) -> dict:
        meta_ipt = task.get_meta_input()
        weights = self.tn.twm(meta_ipt["time_perf"])

        weight_s = pd.Series(weights.detach().cpu().numpy(), index=task.meta_info.columns)
        task = copy.copy(task.task)  # NOTE: this is a shallow copy.
        task["reweighter"] = TimeReweighter(weight_s)
        return task

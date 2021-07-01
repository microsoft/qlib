# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
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

from .utils import fill_diagnal, convert_data_to_tensor, ICLoss
from .dataset import MetaDatasetHDS
from .net import PredNet


class MetaModelDS(MetaTaskModel):
    """
    The meta-model for meta-learning-based data selection.
    """

    def __init__(
        self,
        hist_n=30,
        clip_method="tanh",
        clip_weight=2.0,
        criterion="ic_loss",
        lr=0.0001,
        max_epoch=150,
    ):
        self.hist_n = hist_n
        self.clip_method = clip_method
        self.clip_weight = clip_weight
        self.criterion = criterion
        self.lr = lr
        self.max_epoch = max_epoch
        self.fitted = False

    def fit(self, meta_dataset: MetaDatasetHDS):
        """
        The meta-learning-based data selection interacts directly with meta-dataset due to the close-form proxy measurement.

        Parameters
        ----------
        meta_dataset : MetaDatasetHDS
            The meta-model takes the meta-dataset for its training process.
        """
        recorder = R.get_recorder()
        if not self.fitted:
            for k in set(["lr", "hist_n", "clip_method", "clip_weight", "criterion", "max_epoch"]):
                recorder.log_params(**{k: getattr(self, k)})

        # Training begins
        meta_tasks = meta_dataset.prepare_tasks(["train", "test"])
        num2phase = {0: "train", 1: "test"}
        phase2num = dict(zip(num2phase.values(), num2phase.keys()))
        train_step = 0
        self.tn = PredNet(hist_n=self.hist_n, clip_weight=self.clip_weight, clip_method=self.clip_method)
        opt = optim.Adam(self.tn.parameters(), lr=self.lr)
        loss_l = {}
        for epoch in tqdm(range(self.max_epoch), desc="epoch"):
            for phase, task_list in enumerate(meta_tasks):
                if phase == phase2num["train"]:  # phase 0 for training, 1 for inference
                    self.tn.train()
                    torch.set_grad_enabled(True)
                else:
                    self.tn.eval()
                    torch.set_grad_enabled(False)
                running_loss = 0.0
                pred_y_all = []
                for task in tqdm(task_list, desc=f"{num2phase[phase]} Task", leave=False):
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
                    ) = task.prepare_task_data()
                    pred, weights = self.tn(X, y, time_perf, time_belong, X_test)
                    if self.criterion == "mse":
                        criterion = nn.MSELoss()
                        loss = criterion(pred, y_test)
                    elif self.criterion == "ic_loss":
                        criterion = ICLoss()
                        loss = criterion(pred, y_test, test_idx)

                    if phase == phase2num["train"]:
                        opt.zero_grad()
                        norm_loss = nn.MSELoss()
                        loss.backward()
                        opt.step()
                        train_step += 1
                    elif phase == phase2num["test"]:
                        pass  # pass, leave the work for the inference function
                    #                             self.reweighters[test_period] = SampleReweighter(
                    #                                 pd.Series(weights.detach().cpu().numpy(), index=train_idx)
                    #                             )

                    pred_y_all.append(
                        pd.DataFrame(
                            {
                                "pred": pd.Series(pred.detach().cpu().numpy(), index=test_idx),
                                "label": pd.Series(y_test.detach().cpu().numpy(), index=test_idx),
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

                recorder.log_metrics(**{f"loss/{num2phase[phase]}": running_loss, "step": epoch})
                recorder.log_metrics(**{f"ic/{num2phase[phase]}": ic, "step": epoch})
            recorder.save_objects(**{"model.pkl": self.tn})
        self.fitted = True

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

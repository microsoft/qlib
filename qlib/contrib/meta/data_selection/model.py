# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from qlib.log import get_module_logger
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

from ....data.dataset.weight import Reweighter
from ....model.meta.dataset import MetaTaskDataset
from ....model.meta.model import MetaModel, MetaTaskModel
from ....workflow import R

from .utils import ICLoss
from .dataset import MetaDatasetDS
from qlib.contrib.meta.data_selection.net import PredNet
from qlib.data.dataset.weight import Reweighter
from qlib.log import get_module_logger

logger = get_module_logger("data selection")


class TimeReweighter(Reweighter):
    def __init__(self, time_weight: pd.Series):
        self.time_weight = time_weight

    def reweight(self, data: Union[pd.DataFrame, pd.Series]):
        # TODO: handling TSDataSampler
        w_s = pd.Series(1.0, index=data.index)
        for k, w in self.time_weight.items():
            w_s.loc[slice(*k)] = w
        logger.info(f"Reweighting result: {w_s}")
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
        max_epoch=100,
        seed=43,
    ):
        self.step = step
        self.hist_step_n = hist_step_n
        self.clip_method = clip_method
        self.clip_weight = clip_weight
        self.criterion = criterion
        self.lr = lr
        self.max_epoch = max_epoch
        self.fitted = False
        torch.manual_seed(seed)

    def run_epoch(self, phase, task_list, epoch, opt, loss_l, ignore_weight=False):
        if phase == "train":
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
                ignore_weight=ignore_weight,
            )
            if self.criterion == "mse":
                criterion = nn.MSELoss()
                loss = criterion(pred, meta_input["y_test"])
            elif self.criterion == "ic_loss":
                criterion = ICLoss()
                try:
                    loss = criterion(pred, meta_input["y_test"], meta_input["test_idx"], skip_size=50)
                except ValueError as e:
                    get_module_logger("MetaModelDS").warning(f"Exception `{e}` when calculating IC loss")
                    continue

            assert not np.isnan(loss.detach().item()), "NaN loss!"

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
                        "label": pd.Series(meta_input["y_test"].detach().cpu().numpy(), index=meta_input["test_idx"]),
                    }
                )
            )
            running_loss += loss.detach().item()
        running_loss = running_loss / len(task_list)
        loss_l.setdefault(phase, []).append(running_loss)

        pred_y_all = pd.concat(pred_y_all)
        ic = pred_y_all.groupby("datetime").apply(lambda df: df["pred"].corr(df["label"], method="spearman")).mean()

        R.log_metrics(**{f"loss/{phase}": running_loss, "step": epoch})
        R.log_metrics(**{f"ic/{phase}": ic, "step": epoch})

    def fit(self, meta_dataset: MetaDatasetDS):
        """
        The meta-learning-based data selection interacts directly with meta-dataset due to the close-form proxy measurement.

        Parameters
        ----------
        meta_dataset : MetaDatasetDS
            The meta-model takes the meta-dataset for its training process.
        """

        if not self.fitted:
            for k in set(["lr", "step", "hist_step_n", "clip_method", "clip_weight", "criterion", "max_epoch"]):
                R.log_params(**{k: getattr(self, k)})

        # FIXME: get test tasks for just checking the performance
        phases = ["train", "test"]
        meta_tasks_l = meta_dataset.prepare_tasks(phases)

        if len(meta_tasks_l[1]):
            R.log_params(
                **dict(proxy_test_begin=meta_tasks_l[1][0].task["dataset"]["kwargs"]["segments"]["test"])
            )  # debug: record when the test phase starts

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

    def _prepare_task(self, task: MetaTask) -> dict:
        meta_ipt = task.get_meta_input()
        weights = self.tn.twm(meta_ipt["time_perf"])

        weight_s = pd.Series(weights.detach().cpu().numpy(), index=task.meta_info.columns)
        task = copy.copy(task.task)  # NOTE: this is a shallow copy.
        task["reweighter"] = TimeReweighter(weight_s)
        return task

    def inference(self, meta_dataset: MetaTaskDataset) -> List[dict]:
        res = []
        for mt in meta_dataset.prepare_tasks("test"):
            res.append(self._prepare_task(mt))
        return res

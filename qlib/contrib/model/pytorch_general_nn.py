# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import division
from __future__ import print_function

from torch.utils.data import DataLoader


import numpy as np
import pandas as pd
from typing import Union
import copy

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from qlib.data.dataset.weight import Reweighter

from .pytorch_utils import count_parameters, get_device
from ...model.base import Model
from ...data.dataset import DatasetH, TSDatasetH
from ...data.dataset.handler import DataHandlerLP
from ...utils import (
    init_instance_by_config,
    get_or_create_path,
)
from ...log import get_module_logger

from ...model.utils import ConcatDataset


class GeneralPTNN(Model):
    """
    Motivation:
        We want to provide a Qlib General Pytorch Model Adaptor
        You can reuse it for all kinds of Pytorch models.
        It should include the training and predict process

    Parameters
    ----------
    d_feat : int
        input dimension for each time step
    metric: str
        the evaluation metric used in early stop
    optimizer : str
        optimizer name
    GPU : str
        the GPU ID(s) used for training
    """

    def __init__(
        self,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2000,
        early_stop=20,
        loss="mse",
        weight_decay=0.0,
        optimizer="adam",
        n_jobs=10,
        GPU=0,
        seed=None,
        pt_model_uri="qlib.contrib.model.pytorch_gru_ts.GRUModel",
        pt_model_kwargs={
            "d_feat": 6,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.0,
        },
    ):
        # Set logger.
        self.logger = get_module_logger("GeneralPTNN")
        self.logger.info("GeneralPTNN pytorch version...")

        # set hyper-parameters.
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.weight_decay = weight_decay
        self.device = get_device(GPU)
        self.n_jobs = n_jobs
        self.seed = seed

        self.pt_model_uri, self.pt_model_kwargs = pt_model_uri, pt_model_kwargs
        self.dnn_model = init_instance_by_config({"class": pt_model_uri, "kwargs": pt_model_kwargs})

        self.logger.info(
            "GeneralPTNN parameters setting:"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\ndevice : {}"
            "\nn_jobs : {}"
            "\nuse_GPU : {}"
            "\nweight_decay : {}"
            "\nseed : {}"
            "\npt_model_uri: {}"
            "\npt_model_kwargs: {}".format(
                n_epochs,
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                loss,
                self.device,
                n_jobs,
                self.use_gpu,
                weight_decay,
                seed,
                pt_model_uri,
                pt_model_kwargs,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.logger.info("model:\n{:}".format(self.dnn_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.dnn_model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.dnn_model.parameters(), lr=self.lr, weight_decay=weight_decay)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.dnn_model.parameters(), lr=self.lr, weight_decay=weight_decay)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        # === ReduceLROnPlateau learning rate scheduler ===
        self.lr_scheduler = ReduceLROnPlateau(
            self.train_optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6, threshold=1e-5
        )
        self.fitted = False
        self.dnn_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label, weight):
        loss = weight * (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label, weight=None):
        mask = ~torch.isnan(label)

        if weight is None:
            weight = torch.ones_like(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask].view(-1, 1), weight[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def _get_fl(self, data: torch.Tensor):
        """
        get feature and label from data
        - Handle the different data shape of time series and tabular data

        Parameters
        ----------
        data : torch.Tensor
            input data which maybe 3 dimension or 2 dimension
            - 3dim: [batch_size, time_step, feature_dim]
            - 2dim: [batch_size, feature_dim]

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
        """
        if data.dim() == 3:
            # it is a time series dataset
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)
        elif data.dim() == 2:
            # it is a tabular dataset
            feature = data[:, 0:-1].to(self.device)
            label = data[:, -1].to(self.device)
        else:
            raise ValueError("Unsupported data shape.")
        return feature, label

    def train_epoch(self, data_loader):
        self.dnn_model.train()

        for data, weight in data_loader:
            feature, label = self._get_fl(data)

            pred = self.dnn_model(feature.float())
            loss = self.loss_fn(pred, label, weight.to(self.device))

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.dnn_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        self.dnn_model.eval()

        scores = []
        losses = []

        for data, weight in data_loader:
            feature, label = self._get_fl(data)

            with torch.no_grad():
                pred = self.dnn_model(feature.float())
                loss = self.loss_fn(pred, label, weight.to(self.device))
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset: Union[DatasetH, TSDatasetH],
        evals_result=dict(),
        save_path=None,
        reweighter=None,
    ):
        ists = isinstance(dataset, TSDatasetH)  # is this time series dataset

        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        self.logger.info(f"Train samples: {len(dl_train)}")
        self.logger.info(f"Valid samples: {len(dl_valid)}")
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        if reweighter is None:
            wl_train = np.ones(len(dl_train))
            wl_valid = np.ones(len(dl_valid))
        elif isinstance(reweighter, Reweighter):
            wl_train = reweighter.reweight(dl_train)
            wl_valid = reweighter.reweight(dl_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        # Preprocess for data.  To align to Dataset Interface for DataLoader
        if ists:
            dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
            dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        else:
            # If it is a tabular, we convert the dataframe to numpy to be indexable by DataLoader
            dl_train = dl_train.values
            dl_valid = dl_valid.values

        train_loader = DataLoader(
            ConcatDataset(dl_train, wl_train),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_jobs,
            drop_last=True,
        )
        valid_loader = DataLoader(
            ConcatDataset(dl_valid, wl_valid),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_jobs,
            drop_last=True,
        )
        del dl_train, dl_valid, wl_train, wl_valid

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("Epoch%d: train %.6f, valid %.6f" % (step, train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            # current_lr = self.train_optimizer.param_groups[0]["lr"]
            # self.logger.info("Current learning rate: %.6e" % current_lr)

            self.lr_scheduler.step(val_score)

            if step == 0:
                best_param = copy.deepcopy(self.dnn_model.state_dict())
            if val_score < best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.dnn_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d epoch" % (best_score, best_epoch))
        self.dnn_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(
        self,
        dataset: Union[DatasetH, TSDatasetH],
        batch_size=None,
        n_jobs=None,
    ):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        self.logger.info(f"Test samples: {len(dl_test)}")

        if isinstance(dataset, TSDatasetH):
            dl_test.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
            index = dl_test.get_index()
        else:
            # If it is a tabular, we convert the dataframe to numpy to be indexable by DataLoader
            index = dl_test.index
            dl_test = dl_test.values

        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.dnn_model.eval()
        preds = []

        for data in test_loader:
            feature, _ = self._get_fl(data)
            feature = feature.to(self.device)

            with torch.no_grad():
                pred = self.dnn_model(feature.float()).detach().cpu().numpy()

            preds.append(pred)

        preds_concat = np.concatenate(preds)
        if preds_concat.ndim != 1:
            preds_concat = preds_concat.ravel()

        return pd.Series(preds_concat, index=index)

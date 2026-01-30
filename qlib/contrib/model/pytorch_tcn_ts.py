# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import copy
from ...utils import get_or_create_path
from ...log import get_module_logger

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset.handler import DataHandlerLP
from .tcn import TemporalConvNet


class TCN(Model):
    """TCN Model

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
        d_feat=6,
        n_chans=128,
        kernel_size=5,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2000,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        n_jobs=10,
        GPU=0,
        seed=None,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("TCN")
        self.logger.info("TCN pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.n_chans = n_chans
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.n_jobs = n_jobs
        self.seed = seed

        self.logger.info(
            f"TCN parameters setting:\nd_feat : {d_feat}\nn_chans : {n_chans}\nkernel_size : {kernel_size}\nnum_layers : {num_layers}\ndropout : {dropout}\nn_epochs : {n_epochs}\nlr : {lr}\nmetric : {metric}\nbatch_size : {batch_size}\nearly_stop : {early_stop}\noptimizer : {optimizer.lower()}\nloss_type : {loss}\ndevice : {self.device}\nn_jobs : {n_jobs}\nuse_GPU : {self.use_gpu}\nseed : {seed}"
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.TCN_model = TCNModel(
            num_input=self.d_feat,
            output_size=1,
            num_channels=[self.n_chans] * self.num_layers,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )
        self.logger.info(f"model:\n{self.TCN_model:}")
        self.logger.info(f"model size: {count_parameters(self.TCN_model):.4f} MB")

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.TCN_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.TCN_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(f"optimizer {optimizer} is not supported!")

        self.fitted = False
        self.TCN_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])

        raise ValueError(f"unknown loss `{self.loss}`")

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError(f"unknown metric `{self.metric}`")

    def train_epoch(self, data_loader):
        self.TCN_model.train()

        for data in data_loader:
            data = torch.transpose(data, 1, 2)
            feature = data[:, 0:-1, :].to(self.device)
            label = data[:, -1, -1].to(self.device)

            pred = self.TCN_model(feature.float())
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.TCN_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        self.TCN_model.eval()

        scores = []
        losses = []

        for data in data_loader:
            data = torch.transpose(data, 1, 2)
            feature = data[:, 0:-1, :].to(self.device)
            # feature[torch.isnan(feature)] = 0
            label = data[:, -1, -1].to(self.device)

            with torch.no_grad():
                pred = self.TCN_model(feature.float())
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset,
        evals_result=None,
        save_path=None,
    ):
        if evals_result is None:
            evals_result = {}
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)

        # process nan brought by dataloader
        dl_train.config(fillna_type="ffill+bfill")
        # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")

        train_loader = DataLoader(
            dl_train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_jobs, drop_last=True
        )
        valid_loader = DataLoader(
            dl_valid, batch_size=self.batch_size, shuffle=False, num_workers=self.n_jobs, drop_last=True
        )

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        best_score = -np.inf
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
            _train_loss, train_score = self.test_epoch(train_loader)
            _val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info(f"train {train_score:.6f}, valid {val_score:.6f}")
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.TCN_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info(f"best score: {best_score:.6f} @ {best_epoch}")
        self.TCN_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.TCN_model.eval()
        preds = []

        for data in test_loader:
            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                pred = self.TCN_model(feature.float()).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())


class TCNModel(nn.Module):
    def __init__(self, num_input, output_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.num_input = num_input
        self.tcn = TemporalConvNet(num_input, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        output = self.tcn(x)
        output = self.linear(output[:, :, -1])
        return output.squeeze()

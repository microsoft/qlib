# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import copy
import random
from sklearn.metrics import roc_auc_score, mean_squared_error
import logging
from ...utils import (
    unpack_archive_with_buffer,
    save_multiple_parts_file,
    get_or_create_path,
    drop_nan_by_y_index,
)
from ...log import get_module_logger, TimeInspector

import torch
import torch.nn as nn
import torch.optim as optim

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP


class MTL(Model):
    """MTL Model

    Parameters
    ----------
    d_feat : int
        input dimension for each time step
    metric: str
        the evaluate metric used in early stop
    optimizer : str
        optimizer name
    GPU : str
        the GPU ID(s) used for training
    """

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        batch_size=2000,
        early_stop=20,
        loss="mse",
        fore_optimizer="adam",
        lr=5e-4,
        output_dim=5,
        GPU=0,
        seed=0,
        target_label=0,
        mode='hard',
        lowest_valid_performance=0.993,
        **kwargs
    ):
        # Set logger.
        self.logger = get_module_logger("MTL")
        self.logger.info("MTL pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.loss = loss
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() else "cpu")
        self.use_gpu = torch.cuda.is_available()
        self.seed = seed
        self.output_dim = output_dim
        self.target_label = target_label
        self.mode = mode
        self.lowest_valid_performance = lowest_valid_performance
        self._fore_optimizer = fore_optimizer
        self.lr = lr

        self.logger.info(
            "MTL parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\ntarget_label : {}"
            "\nlr : {}"
            "\nmode : {}"
            "\nloss_type : {}"
            "\nvisible_GPU : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                batch_size,
                early_stop,
                target_label,
                lr, 
                mode,
                loss,
                GPU,
                self.use_gpu,
                seed,
            )
        )

    def loss_fn(self, pred, label):

        if self.mode == 'hard':
            loc = np.random.randint(0, label.shape[1], label.shape[0])
            loss = (pred - label[np.arange(label.shape[0]), loc]) ** 2
            return torch.mean(loss)

        elif self.mode == 'soft':
            loss = (pred - label.transpose(0,1)) ** 2
            return torch.mean(loss)

        else:
            raise NotImplementedError("mode {} is not supported!".format(self.mode))

    def train_epoch(self, x_train, y_train):

        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        self.fore_model.train()

        for i in range(len(indices))[:: self.batch_size]:

            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            pred = self.fore_model(feature)
            loss = self.loss_fn(pred, label)  

            self.fore_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.fore_model.parameters(), 3.0)
            self.fore_optimizer.step()


    def test_epoch(self, data_x, data_y):

        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.fore_model.eval()

        losses = []

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:

            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)

            pred = self.fore_model(feature)
            loss = torch.mean((pred - label[:, abs(self.target_label)]) ** 2)
            losses.append(loss.item())

        return np.mean(losses)

    def fit(
        self,
        dataset: DatasetH,
        verbose=True,
        save_path=None,
    ):
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]
        x_test, y_test = df_test["feature"], df_test["label"]

        if save_path == None:
            save_path = get_or_create_path(save_path)
        best_loss = np.inf
        while best_loss > self.lowest_valid_performance:
            if best_loss < np.inf:
                print("Failed! Start retraining.")
                self.seed = random.randint(0, 1000)  # reset random seed

            if self.seed is not None:
                np.random.seed(self.seed)
                torch.manual_seed(self.seed)

            best_loss = self.training(
                x_train, y_train, x_valid, y_valid, x_test, y_test, verbose=verbose, save_path=save_path
            )

    def training(
        self,
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
        verbose=True,
        save_path=None,
    ):

        self.fore_model = GRUModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        if self._fore_optimizer.lower() == "adam":
            self.fore_optimizer = optim.Adam(self.fore_model.parameters(), lr=self.lr)
        elif self._fore_optimizer.lower() == "gd":
            self.fore_optimizer = optim.SGD(self.fore_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(self._fore_optimizer))

        self.fitted = False
        self.fore_model.to(self.device)

        best_loss = np.inf
        best_epoch = 0
        stop_round = 0

        for epoch in range(self.n_epochs):
            print("Epoch:", epoch)

            print("training...")
            self.train_epoch(x_train, y_train)
            print("evaluating...")
            val_loss = self.test_epoch(x_valid, y_valid)
            test_loss = self.test_epoch(x_test, y_test)

            if verbose:
                print("valid %.6f, test %.6f" % (val_loss, test_loss))

            if val_loss < best_loss:
                best_loss = val_loss
                stop_round = 0
                best_epoch = epoch
                torch.save(copy.deepcopy(self.fore_model.state_dict()), save_path + "_fore_model.bin")

            else:
                stop_round += 1
                if stop_round >= self.early_stop:
                    print("early stop")
                    break

        print("best loss:", best_loss, "@", best_epoch)
        best_param = torch.load(save_path + "_fore_model.bin")
        self.fore_model.load_state_dict(best_param)
        self.fitted = True

        if self.use_gpu:
            torch.cuda.empty_cache()

        return best_loss

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare("test", col_set="feature")
        index = x_test.index
        self.fore_model.eval()
        x_values = x_test.values
        sample_num = x_values.shape[0]
        preds = []

        for begin in range(sample_num)[:: self.batch_size]:

            if sample_num - begin < self.batch_size:
                end = sample_num
            else:
                end = begin + self.batch_size

            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)

            with torch.no_grad():
                if self.use_gpu:
                    pred = self.fore_model(x_batch).detach().cpu().numpy()
                else:
                    pred = self.fore_model(x_batch).detach().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


class GRUModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat

    def forward(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()

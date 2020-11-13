# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import copy
from sklearn.metrics import roc_auc_score, mean_squared_error
import logging
from ...utils import unpack_archive_with_buffer, save_multiple_parts_file, create_save_path, drop_nan_by_y_index
from ...log import get_module_logger, TimeInspector

import torch
import torch.nn as nn
import torch.optim as optim

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP


class GRU(Model):
    """GRU Model

    Parameters
    ----------
    input_dim : int
        input dimension
    output_dim : int
        output dimension
    layers : tuple
        layer sizes
    lr : float
        learning rate
    lr_decay : float
        learning rate decay
    lr_decay_steps : int
        learning rate decay steps
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
        lr=0.001,
        batch_size=2000,
        early_stop=20,
        eval_steps=5,
        loss="mse",
        lr_decay=0.96,
        lr_decay_steps=100,
        optimizer="gd",
        GPU="0",
        seed=0,
        **kwargs
    ):
        # Set logger.
        self.logger = get_module_logger("GRU")
        self.logger.info("GRU pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.eval_steps = eval_steps
        self.lr_decay = lr_decay
        self.lr_decay_steps = lr_decay_steps
        self.optimizer = optimizer.lower()
        self.loss_type = loss
        self.visible_GPU = GPU
        self.use_gpu = torch.cuda.is_available()
        self.seed = seed

        self.logger.info(
            "GRU parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\neval_steps : {}"
            "\nlr_decay : {}"
            "\nlr_decay_steps : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\nvisible_GPU : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                lr,
                batch_size,
                early_stop,
                eval_steps,
                lr_decay,
                lr_decay_steps,
                optimizer.lower(),
                loss,
                GPU,
                self.use_gpu,
                seed,
            )
        )

        if loss not in {"mse", "binary"}:
            raise NotImplementedError("loss {} is not supported!".format(loss))
        self._scorer = mean_squared_error if loss == "mse" else roc_auc_score

        self.gru_model = GRUModel(
            d_feat=self.d_feat, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout
        )
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.gru_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.gru_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        # Reduce learning rate when loss has stopped decrease
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.train_optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            verbose=True,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0.00001,
            eps=1e-08,
        )

        self._fitted = False
        if self.use_gpu:
            self.gru_model.cuda()
            # set the visible GPU
            if self.visible_GPU:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.visible_GPU

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        verbose=True,
        save_path=None,
    ):

        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        # Lightgbm need 1D array as its label
        save_path = create_save_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_loss = np.inf
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self._fitted = True
        # return
        # prepare training data
        x_train_values = torch.from_numpy(x_train.values).float()
        y_train_values = torch.from_numpy(np.squeeze(y_train.values)).float()
        train_num = y_train_values.shape[0]

        # prepare validation data
        x_val_auto = torch.from_numpy(x_valid.values).float()
        y_val_auto = torch.from_numpy(np.squeeze(y_valid.values)).float()

        if self.use_gpu:
            x_val_auto = x_val_auto.cuda()
            y_val_auto = y_val_auto.cuda()

        for step in range(self.n_epochs):
            if stop_steps >= self.early_stop:
                if verbose:
                    self.logger.info("\tearly stop")
                break
            loss = AverageMeter()
            self.gru_model.train()
            self.train_optimizer.zero_grad()

            choice = np.random.choice(train_num, self.batch_size)
            x_batch_auto = x_train_values[choice]
            y_batch_auto = y_train_values[choice]

            if self.use_gpu:
                x_batch_auto = x_batch_auto.float().cuda()
                y_batch_auto = y_batch_auto.float().cuda()

            # forward
            preds = self.gru_model(x_batch_auto)
            cur_loss = self.get_loss(preds, y_batch_auto, self.loss_type)
            cur_loss.backward()
            self.train_optimizer.step()
            loss.update(cur_loss.item())

            # validation
            train_loss += loss.val
            # print(loss.val)
            if step and step % self.eval_steps == 0:
                stop_steps += 1
                train_loss /= self.eval_steps

                with torch.no_grad():
                    self.gru_model.eval()
                    loss_val = AverageMeter()

                    # forward
                    preds = self.gru_model(x_val_auto)
                    cur_loss_val = self.get_loss(preds, y_val_auto, self.loss_type)
                    loss_val.update(cur_loss_val.item())

                if verbose:
                    self.logger.info(
                        "[Epoch {}]: train_loss {:.6f}, valid_loss {:.6f}".format(step, train_loss, loss_val.val)
                    )
                evals_result["train"].append(train_loss)
                evals_result["valid"].append(loss_val.val)
                if loss_val.val < best_loss:
                    if verbose:
                        self.logger.info(
                            "\tvalid loss update from {:.6f} to {:.6f}, save checkpoint.".format(
                                best_loss, loss_val.val
                            )
                        )
                    best_loss = loss_val.val
                    stop_steps = 0
                    torch.save(self.gru_model.state_dict(), save_path)
                train_loss = 0
                # update learning rate
                self.scheduler.step(cur_loss_val)

        # restore the optimal parameters after training ??
        # self.gru_model.load_state_dict(torch.load(save_path))
        if self.use_gpu:
            torch.cuda.empty_cache()

    def get_loss(self, pred, target, loss_type):
        if loss_type == "mse":
            sqr_loss = (pred - target) ** 2
            loss = sqr_loss.mean()
            return loss
        elif loss_type == "binary":
            loss = nn.BCELoss()
            return loss(pred, target)
        else:
            raise NotImplementedError("loss {} is not supported!".format(loss_type))

    def predict(self, dataset):
        if not self._fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare("test", col_set="feature")
        index = x_test.index
        x_test = torch.from_numpy(x_test.values).float()

        if self.use_gpu:
            x_test = x_test.cuda()
        self.gru_model.eval()

        with torch.no_grad():
            if self.use_gpu:
                preds = self.gru_model(x_test).detach().cpu().numpy()
            else:
                preds = self.gru_model(x_test).detach().numpy()
        return pd.Series(preds, index=index)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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

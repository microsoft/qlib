# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import division
from __future__ import print_function

from torch.utils.data import DataLoader, RandomSampler, StackDataset


import os
import numpy as np
import pandas as pd
from typing import Callable, Optional, Text, Union
from sklearn.metrics import roc_auc_score, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import StackDataset

from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset import DatasetH, TSDatasetH
from ...data.dataset.handler import DataHandlerLP
from ...utils import (
    auto_filter_kwargs,
    init_instance_by_config,
    unpack_archive_with_buffer,
    save_multiple_parts_file,
    get_or_create_path,
)
from ...log import get_module_logger
from ...workflow import R
from qlib.contrib.meta.data_selection.utils import ICLoss
from torch.nn import DataParallel


class GeneralPTNN(Model):
    """General Pytorch Neural Network Model
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
    optimizer : str
        optimizer name
    GPU : int
        the GPU ID used for training
    """

    def __init__(
        self,
        lr=0.001,
        max_steps=300,
        batch_size=2000,
        early_stop_rounds=50,
        eval_steps=20,
        optimizer="gd",
        loss="mse",
        GPU=0,
        seed=None,
        weight_decay=0.0,
        data_parall=False,
        scheduler: Optional[Union[Callable]] = "default",  # when it is Callable, it accept one argument named optimizer
        init_model=None,
        eval_train_metric=False,
        pt_model_uri="qlib.contrib.model.pytorch_nn.Net",
        pt_model_kwargs={
            "input_dim": 360,
            "layers": (256,),
        },
        valid_key=DataHandlerLP.DK_L,
        # TODO: Infer Key is a more reasonable key. But it requires more detailed processing on label processing
    ):
        # Set logger.
        self.logger = get_module_logger("DNNModelPytorch")
        self.logger.info("DNN pytorch version...")

        # set hyper-parameters.
        self.lr = lr
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.early_stop_rounds = early_stop_rounds
        self.eval_steps = eval_steps
        self.optimizer = optimizer.lower()
        self.loss_type = loss
        if isinstance(GPU, str):
            self.device = torch.device(GPU)
        else:
            self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.weight_decay = weight_decay
        self.data_parall = data_parall
        self.eval_train_metric = eval_train_metric
        self.valid_key = valid_key

        self.best_step = None

        self.logger.info(
            "DNN parameters setting:"
            f"\nlr : {lr}"
            f"\nmax_steps : {max_steps}"
            f"\nbatch_size : {batch_size}"
            f"\nearly_stop_rounds : {early_stop_rounds}"
            f"\neval_steps : {eval_steps}"
            f"\noptimizer : {optimizer}"
            f"\nloss_type : {loss}"
            f"\nseed : {seed}"
            f"\ndevice : {self.device}"
            f"\nuse_GPU : {self.use_gpu}"
            f"\nweight_decay : {weight_decay}"
            f"\nenable data parall : {self.data_parall}"
            f"\npt_model_uri: {pt_model_uri}"
            f"\npt_model_kwargs: {pt_model_kwargs}"
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        if loss not in {"mse", "binary"}:
            raise NotImplementedError("loss {} is not supported!".format(loss))
        self._scorer = mean_squared_error if loss == "mse" else roc_auc_score

        if init_model is None:
            self.dnn_model = init_instance_by_config({"class": pt_model_uri, "kwargs": pt_model_kwargs})

            if self.data_parall:
                self.dnn_model = DataParallel(self.dnn_model).to(self.device)
        else:
            self.dnn_model = init_model

        self.logger.info("model:\n{:}".format(self.dnn_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.dnn_model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.dnn_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.dnn_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        if scheduler == "default":
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
        elif scheduler is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler(optimizer=self.train_optimizer)

        self.fitted = False
        self.dnn_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    
    def _eval_valid_dl(self, valid_loader, val_index):
        with torch.no_grad():
            self.dnn_model.eval()
            val_loss = []
            val_pred = []
            val_label = []
            for x_batch, y_batch in valid_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                cur_loss = self.get_loss(preds, y_batch, self.loss_type)
                val_loss.append(cur_loss.detach().cpu().numpy().item())
            val_loss = np.mean(val_loss)
            val_pred = torch.cat(val_pred, axis=0).detach().cpu().numpy()
            val_label = torch.cat(val_label, axis=0).detach().cpu().numpy()
            val_metric = self.get_metric(val_pred, val_label, val_index).detach().cpu().numpy().item()
        return val_loss, val_metric

    def fit(
        self,
        dataset: Union[DatasetH, TSDatasetH],
        verbose=True,
        save_path=None,
    ):

        ists = isinstance(dataset, TSDatasetH)  # is this time series dataset

        # prepare training
        train_x = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
        train_y = dataset.prepare("train", col_set="label", data_key=DataHandlerLP.DK_L)
        train_ds = StackDataset(train_x, train_y)
        train_sampler = RandomSampler(train_ds)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, sampler=train_sampler)

        # prepare validation
        valid_x = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
        valid_y = dataset.prepare("train", col_set="label", data_key=DataHandlerLP.DK_L)
        valid_ds = StackDataset(valid_x, valid_y)
        valid_loader = DataLoader(valid_ds, batch_size=self.batch_size, shuffle=False)
        if ists:
            val_index = valid_x.data_index
        else:
            val_index = valid_x.index


        save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_loss = np.inf
        # train
        self.logger.info("training...")


        for step in range(1, self.max_steps + 1):
            if stop_steps >= self.early_stop_rounds:
                if verbose:
                    self.logger.info("\tearly stop")
                break
            loss = AverageMeter()
            self.dnn_model.train()
            self.train_optimizer.zero_grad()

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # forward
                preds = self.dnn_model(x_batch)
                cur_loss = self.get_loss(preds, y_batch, self.loss_type)
                cur_loss.backward()
                self.train_optimizer.step()
                loss.update(cur_loss.item())
                R.log_metrics(train_loss=loss.avg, step=step)

            # validation
            train_loss += loss.val
            # for every `eval_steps` steps or at the last steps, we will evaluate the model.
            if step % self.eval_steps == 0 or step == self.max_steps:
                stop_steps += 1
                train_loss /= self.eval_steps

                val_loss, val_metric = self._eval_valid_dl(valid_loader, val_index)
                R.log_metrics(val_loss=val_loss, step=step)
                R.log_metrics(val_metric=val_metric, step=step)

                if val_loss < best_loss:
                    if verbose:
                        self.logger.info(
                            "\tvalid loss update from {:.6f} to {:.6f}, save checkpoint.".format(
                                best_loss, val_loss
                            )
                        )
                    best_loss = val_loss
                    self.best_step = step
                    R.log_metrics(best_step=self.best_step, step=step)
                    stop_steps = 0
                    torch.save(self.dnn_model.state_dict(), save_path)
                train_loss = 0
                # update learning rate
                if self.scheduler is not None:
                    auto_filter_kwargs(self.scheduler.step, warning=False)(metrics=val_loss, epoch=step)
                R.log_metrics(lr=self.get_lr(), step=step)

            # restore the optimal parameters after training
            self.dnn_model.load_state_dict(torch.load(save_path, map_location=self.device))
        if self.use_gpu:
            torch.cuda.empty_cache()

    def get_lr(self):
        assert len(self.train_optimizer.param_groups) == 1
        return self.train_optimizer.param_groups[0]["lr"]

    def get_loss(self, pred, target, loss_type, w=None):
        pred, target = pred.reshape(-1), target.reshape(-1)
        if w is None:
            # make it ones and the same size with pred
            w = torch.ones_like(pred).to(pred.device)

        if loss_type == "mse":
            sqr_loss = torch.mul(pred - target, pred - target)
            loss = torch.mul(sqr_loss, w).mean()
            return loss
        elif loss_type == "binary":
            loss = nn.BCEWithLogitsLoss(weight=w)
            return loss(pred, target)
        else:
            raise NotImplementedError("loss {} is not supported!".format(loss_type))

    def get_metric(self, pred, target, index):
        # NOTE: the order of the index must follow <datetime, instrument> sorted order
        return -ICLoss()(pred, target, index)  # pylint: disable=E1130

    def _nn_predict(self, data, return_cpu=True):
        """Reusing predicting NN.
        Scenarios
        1) test inference (data may come from CPU and expect the output data is on CPU)
        2) evaluation on training (data may come from GPU)
        """
        if not isinstance(data, torch.Tensor):
            if isinstance(data, pd.DataFrame):
                data = data.values
            data = torch.Tensor(data)
        data = data.to(self.device)
        preds = []
        self.dnn_model.eval()
        with torch.no_grad():
            batch_size = 8096
            for i in range(0, len(data), batch_size):
                x = data[i : i + batch_size]
                preds.append(self.dnn_model(x.to(self.device)).detach().reshape(-1))
        if return_cpu:
            preds = np.concatenate([pr.cpu().numpy() for pr in preds])
        else:
            preds = torch.cat(preds, axis=0)
        return preds

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        x_test_pd = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        preds = self._nn_predict(x_test_pd)
        return pd.Series(preds.reshape(-1), index=x_test_pd.index)

    def save(self, filename, **kwargs):
        with save_multiple_parts_file(filename) as model_dir:
            model_path = os.path.join(model_dir, os.path.split(model_dir)[-1])
            # Save model
            torch.save(self.dnn_model.state_dict(), model_path)

    def load(self, buffer, **kwargs):
        with unpack_archive_with_buffer(buffer) as model_dir:
            # Get model name
            _model_name = os.path.splitext(list(filter(lambda x: x.startswith("model.bin"), os.listdir(model_dir)))[0])[
                0
            ]
            _model_path = os.path.join(model_dir, _model_name)
            # Load model
            self.dnn_model.load_state_dict(torch.load(_model_path, map_location=self.device))
        self.fitted = True


class AverageMeter:
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

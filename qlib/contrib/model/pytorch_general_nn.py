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
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import StackDataset

from qlib.data.dataset.weight import Reweighter

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
                preds = self.dnn_model(x_batch)
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
        x_test_pd = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        preds = self._nn_predict(x_test_pd)
        return pd.Series(preds.reshape(-1), index=x_test_pd.index)


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
        optimizer="adam",
        n_jobs=10,
        GPU=0,
        seed=None,
        pt_model_uri="qlib.contrib.model.pytorch_gru_ts.GRUModel",
        pt_model_kwargs={
            "d_feat":6,
            "hidden_size":64,
            "num_layers":2,
            "dropout":0.,
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
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
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
            self.train_optimizer = optim.Adam(self.dnn_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.dnn_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

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
            return self.mse(pred[mask], label[mask], weight[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

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
            feature , label = self._get_fl(data)

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
            feature , label = self._get_fl(data)

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
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if step == 0:
                best_param = copy.deepcopy(self.dnn_model.state_dict())
            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.dnn_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.dnn_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: Union[DatasetH, TSDatasetH]):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)

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
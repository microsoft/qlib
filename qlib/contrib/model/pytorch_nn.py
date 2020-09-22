# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error
import logging
from ...utils import unpack_archive_with_buffer, save_multiple_parts_file, create_save_path, drop_nan_by_y_index
from ...log import get_module_logger, TimeInspector

import torch
import torch.nn as nn
import torch.optim as optim

from .base import Model


class DNNModelPytorch(Model):
    """DNN Model

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
        input_dim,
        output_dim,
        layers=(256, 256, 128),
        lr=0.001,
        max_steps=300,
        batch_size=2000,
        early_stop_rounds=50,
        eval_steps=20,
        lr_decay=0.96,
        lr_decay_steps=100,
        optimizer="gd",
        loss="mse",
        GPU="0",
        **kwargs
    ):
        # Set logger.
        self.logger = get_module_logger("DNNModelPytorch")
        self.logger.info("DNN pytorch version...")

        # set hyper-parameters.
        self.layers = layers
        self.lr = lr
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.early_stop_rounds = early_stop_rounds
        self.eval_steps = eval_steps
        self.lr_decay = lr_decay
        self.lr_decay_steps = lr_decay_steps
        self.optimizer = optimizer.lower()
        self.loss_type = loss
        self.visible_GPU = GPU

        self.logger.info(
            "DNN parameters setting:"
            "\nlayers : {}"
            "\nlr : {}"
            "\nmax_steps : {}"
            "\nbatch_size : {}"
            "\nearly_stop_rounds : {}"
            "\neval_steps : {}"
            "\nlr_decay : {}"
            "\nlr_decay_steps : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\neval_steps : {}"
            "\nvisible_GPU : {}".format(
                layers,
                lr,
                max_steps,
                batch_size,
                early_stop_rounds,
                eval_steps,
                lr_decay,
                lr_decay_steps,
                optimizer,
                loss,
                eval_steps,
                GPU,
            )
        )

        if loss not in {"mse", "binary"}:
            raise NotImplementedError("loss {} is not supported!".format(loss))
        self._scorer = mean_squared_error if loss == "mse" else roc_auc_score

        self.dnn_model = Net(input_dim, output_dim, layers, loss=self.loss_type)
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.dnn_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.dnn_model.parameters(), lr=self.lr)
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
        self.dnn_model.cuda()

        # set the visible GPU
        if self.visible_GPU:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.visible_GPU

    def fit(
        self,
        x_train,
        y_train,
        x_valid,
        y_valid,
        w_train=None,
        w_valid=None,
        evals_result=dict(),
        verbose=True,
        save_path=None,
    ):

        if w_train is None:
            w_train = pd.DataFrame(np.ones_like(y_train.values), index=y_train.index)
        if w_valid is None:
            w_valid = pd.DataFrame(np.ones_like(y_valid.values), index=y_valid.index)

        save_path = create_save_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_loss = np.inf
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self._fitted = True
        #return
        # prepare training data
        x_train_values = torch.from_numpy(x_train.values).float()
        y_train_values = torch.from_numpy(y_train.values).float()
        w_train_values = torch.from_numpy(w_train.values).float()
        train_num = y_train_values.shape[0]

        # prepare validation data
        x_val_cuda = torch.from_numpy(x_valid.values).float()
        y_val_cuda = torch.from_numpy(y_valid.values).float()
        w_val_cuda = torch.from_numpy(w_valid.values).float()

        x_val_cuda = x_val_cuda.cuda()
        y_val_cuda = y_val_cuda.cuda()
        w_val_cuda = w_val_cuda.cuda()

        for step in range(self.max_steps):
            if stop_steps >= self.early_stop_rounds:
                if verbose:
                    self.logger.info("\tearly stop")
                break
            loss = AverageMeter()
            self.dnn_model.train()
            self.train_optimizer.zero_grad()

            choice = np.random.choice(train_num, self.batch_size)
            x_batch = x_train_values[choice]
            y_batch = y_train_values[choice]
            w_batch = w_train_values[choice]

            x_batch_cuda = x_batch.float().cuda()
            y_batch_cuda = y_batch.float().cuda()
            w_batch_cuda = w_batch.float().cuda()

            # forward
            preds = self.dnn_model(x_batch_cuda)
            cur_loss = self.get_loss(preds, w_batch_cuda, y_batch_cuda, self.loss_type)
            cur_loss.backward()
            self.train_optimizer.step()
            loss.update(cur_loss.item())

            # validation
            train_loss += loss.val
            #print(loss.val)
            if step and step % self.eval_steps == 0:
                stop_steps += 1
                train_loss /= self.eval_steps

                with torch.no_grad():
                    self.dnn_model.eval()
                    loss_val = AverageMeter()

                    # forward
                    preds = self.dnn_model(x_val_cuda)
                    cur_loss_val = self.get_loss(preds, w_val_cuda, y_val_cuda, self.loss_type)
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
                    torch.save(self.dnn_model.state_dict(), save_path)
                train_loss = 0
                # update learning rate
                self.scheduler.step(cur_loss_val)

        # restore the optimal parameters after training ??
        self.dnn_model.load_state_dict(torch.load(save_path))
        torch.cuda.empty_cache()

    def get_loss(self, pred, w, target, loss_type):
        if loss_type == "mse":
            sqr_loss = torch.mul(pred - target, pred - target)
            loss = torch.mul(sqr_loss, w).mean()
            return loss
        elif loss_type == "binary":
            loss = nn.BCELoss()
            return loss(pred, target)
        else:
            raise NotImplementedError("loss {} is not supported!".format(loss_type))

    def predict(self, x_test):
        if not self._fitted:
            raise ValueError("model is not fitted yet!")
        x_test = torch.from_numpy(x_test.values).float().cuda()
        self.dnn_model.eval()
        
        with torch.no_grad():
            preds = self.dnn_model(x_test).detach().cpu().numpy()
        return preds

    def score(self, x_test, y_test, w_test=None):
        # Remove rows from x, y and w, which contain Nan in any columns in y_test.
        x_test, y_test, w_test = drop_nan_by_y_index(x_test, y_test, w_test)
        preds = self.predict(x_test)
        w_test_weight = None if w_test is None else w_test.values
        return self._scorer(y_test.values, preds, sample_weight=w_test_weight)

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
            self.dnn_model.load_state_dict(torch.load(_model_path))
        self._fitted = True

    def finetune(self, x_train, y_train, x_valid, y_valid, w_train=None, w_valid=None, **kwargs):
        self.fit(x_train, y_train, x_valid, y_valid, w_train=w_train, w_valid=w_valid, **kwargs)


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


class Net(nn.Module):
    def __init__(self, input_dim, output_dim, layers=(256, 256, 256), loss="mse"):
        super(Net, self).__init__()
        layers = [input_dim] + list(layers)
        dnn_layers = []
        drop_input = nn.Dropout(0.1)
        dnn_layers.append(drop_input)
        for i, (input_dim, hidden_units) in enumerate(zip(layers[:-1], layers[1:])):
            fc = nn.Linear(input_dim, hidden_units)
            activation = nn.ReLU()
            bn = nn.BatchNorm1d(hidden_units)
            drop = nn.Dropout(0.1)
            seq = nn.Sequential(fc, bn, activation, drop)
            dnn_layers.append(seq)

        if loss == "mse":
            fc = nn.Linear(hidden_units, output_dim)
            dnn_layers.append(fc)

        elif loss == "binary":
            fc = nn.Linear(hidden_units, output_dim)
            sigmoid = nn.Sigmoid()
            dnn_layers.append(nn.Sequential(fc, sigmoid))
        else:
            raise NotImplementedError("loss {} is not supported!".format(loss))
        # optimizer
        self.dnn_layers = nn.ModuleList(dnn_layers)
        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1)

    def forward(self, x):
        cur_output = x
        for i, now_layer in enumerate(self.dnn_layers):
            cur_output = now_layer(cur_output)
        return cur_output

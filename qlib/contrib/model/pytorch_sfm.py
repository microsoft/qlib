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
import torch.nn.init as init
import torch.optim as optim

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP

class SFM_Model(nn.Module):
    def __init__(self, d_feat=6, output_dim = 1, freq_dim = 10, hidden_size = 64, dropout_W = 0.0, dropout_U = 0.0, device = "cpu"):
        super().__init__()

        self.input_dim  = d_feat
        self.output_dim = output_dim
        self.freq_dim = freq_dim
        self.hidden_dim = hidden_size
        self.device = device

        self.W_i = nn.Parameter(init.xavier_uniform_(torch.empty((self.input_dim, self.hidden_dim))))
        self.U_i = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_i = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_ste = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        self.U_ste = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_ste = nn.Parameter(torch.ones(self.hidden_dim))

        self.W_fre = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.freq_dim)))
        self.U_fre = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.freq_dim)))
        self.b_fre = nn.Parameter(torch.ones(self.freq_dim))

        self.W_c = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        self.U_c = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_c = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_o = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        self.U_o = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_o = nn.Parameter(torch.zeros(self.hidden_dim))

        self.U_a = nn.Parameter(init.orthogonal_(torch.empty(self.freq_dim, 1)))
        self.b_a = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_p = nn.Parameter(init.xavier_uniform_(torch.empty(self.hidden_dim, self.output_dim)))
        self.b_p = nn.Parameter(torch.zeros(self.output_dim))
        
        self.activation = nn.Tanh()
        self.inner_activation = nn.Hardsigmoid()
        self.dropout_W, self.dropout_U = (dropout_W, dropout_U)
        self.fc_out = nn.Linear(self.output_dim, 1)

        self.states = []
    
    def forward(self, input):
        input = input.reshape(len(input), self.input_dim, -1) # [N, F, T]
        input = input.permute(0, 2, 1) # [N, T, F]
        time_step = input.shape[1]
        
        for ts in range(time_step):
            x = input[:, ts,:]
            if(len(self.states)==0): #hasn't initialized yet
                self.init_states(x)
            self.get_constants(x)
            p_tm1 = self.states[0]
            h_tm1 = self.states[1]
            S_re_tm1 = self.states[2]
            S_im_tm1 = self.states[3]
            time_tm1 = self.states[4]
            B_U = self.states[5]
            B_W = self.states[6]
            frequency = self.states[7]

            x_i = torch.matmul(x * B_W[0], self.W_i) + self.b_i
            x_ste = torch.matmul(x * B_W[0], self.W_ste) + self.b_ste
            x_fre = torch.matmul(x * B_W[0], self.W_fre) + self.b_fre
            x_c = torch.matmul(x * B_W[0], self.W_c) + self.b_c
            x_o = torch.matmul(x * B_W[0], self.W_o) + self.b_o
            
            i = self.inner_activation(x_i + torch.matmul(h_tm1 * B_U[0], self.U_i)) # not sure whether I am doing in the right unsquuze
            

            ste = self.inner_activation(x_ste + torch.matmul(h_tm1 * B_U[0], self.U_ste))
            fre = self.inner_activation(x_fre + torch.matmul(h_tm1 * B_U[0], self.U_fre))

            ste = torch.reshape(ste, (-1, self.hidden_dim, 1))
            fre = torch.reshape(fre, (-1, 1, self.freq_dim))
            
            f = ste * fre
            
            c = i * self.activation(x_c + torch.matmul(h_tm1 * B_U[0], self.U_c))

            time = time_tm1 + 1

            omega = torch.tensor(2 * np.pi) * time * frequency

            re = torch.cos(omega) 
            im = torch.sin(omega)
            
            c = torch.reshape(c, (-1, self.hidden_dim, 1))

            S_re = f * S_re_tm1 + c * re
            S_im = f * S_im_tm1 + c * im
            
            A = torch.square(S_re) + torch.square(S_im)

            A = torch.reshape(A, (-1, self.freq_dim)).float()
            A_a = torch.matmul(A * B_U[0], self.U_a)
            A_a = torch.reshape(A_a, (-1, self.hidden_dim))
            a = self.activation(A_a + self.b_a)
            
            o = self.inner_activation(x_o + torch.matmul(h_tm1 * B_U[0], self.U_o))

            h = o * a
            p = torch.matmul(h, self.W_p) + self.b_p

            self.states = [p, h, S_re, S_im, time, None, None, None]
        self.states = []    
        return self.fc_out(p).squeeze()

    def init_states(self, x):
        reducer_f = torch.zeros((self.hidden_dim, self.freq_dim)).to(self.device)
        reducer_p = torch.zeros((self.hidden_dim, self.output_dim)).to(self.device)
        
        init_state_h = torch.zeros(self.hidden_dim).to(self.device)
        init_state_p = torch.matmul(init_state_h, reducer_p)
        
        init_state = torch.zeros_like(init_state_h).to(self.device)
        init_freq = torch.matmul(init_state_h, reducer_f)

        init_state = torch.reshape(init_state, (-1, self.hidden_dim, 1))
        init_freq = torch.reshape(init_freq, (-1, 1, self.freq_dim))
        
        init_state_S_re = init_state * init_freq
        init_state_S_im = init_state * init_freq
        
        init_state_time = torch.tensor(0).to(self.device)

        self.states = [init_state_p, init_state_h, init_state_S_re, init_state_S_im, init_state_time, None, None, None]

    def get_constants(self, x):
        constants = []
        constants.append([torch.tensor(1.).to(self.device) for _ in range(6)])
        constants.append([torch.tensor(1.).to(self.device) for _ in range(7)])
        array = np.array([float(ii)/self.freq_dim for ii in range(self.freq_dim)])
        constants.append(torch.tensor(array).to(self.device))

        self.states[5:] = constants

class SFM(Model):
    """SFM Model

    Parameters
    ----------
    input_dim : int
        input dimension
    output_dim : int
        output dimension
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
        output_dim=1,
        freq_dim = 10,
        dropout_W=0.0,
        dropout_U=0.0,
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
        self.logger = get_module_logger("SFM")
        self.logger.info("SFM pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.freq_dim = freq_dim
        self.dropout_W = dropout_W
        self.dropout_U = dropout_U
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.eval_steps = eval_steps
        self.lr_decay = lr_decay
        self.lr_decay_steps = lr_decay_steps
        self.optimizer = optimizer.lower()
        self.loss_type = loss
        self.device = 'cuda:%d'%(GPU) if torch.cuda.is_available() else 'cpu'
        self.use_gpu = torch.cuda.is_available()
        self.seed = seed

        self.logger.info(
            "SFM parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\noutput_size : {}"
            "\nfrequency_dimension : {}" 
            "\ndropout_W: {}"
            "\ndropout_U: {}"
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
                output_dim,
                freq_dim,
                dropout_W,
                dropout_U,
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

        self.sfm_model = SFM_Model(
            d_feat=self.d_feat, 
            output_dim = self.output_dim,
            hidden_size = self.hidden_size, 
            freq_dim = self.freq_dim, 
            dropout_W=self.dropout_W, 
            dropout_U = self.dropout_U, 
            device = self.device
            )
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.sfm_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.sfm_model.parameters(), lr=self.lr)
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
        self.sfm_model.to(self.device)

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        verbose=True,
        save_path=None,
        **kwargs
    ):

        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        save_path = create_save_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_loss = np.inf
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self._fitted = True

        # prepare training data
        x_train_values = torch.from_numpy(x_train.values).float()
        y_train_values = torch.from_numpy(np.squeeze(y_train.values)).float()
        train_num = y_train_values.shape[0]

        # prepare validation data
        x_val_auto = torch.from_numpy(x_valid.values).float()
        y_val_auto = torch.from_numpy(np.squeeze(y_valid.values)).float()

        x_val_auto = x_val_auto.to(self.device)
        y_val_auto = y_val_auto.to(self.device)

        for step in range(self.n_epochs):
            if stop_steps >= self.early_stop:
                if verbose:
                    self.logger.info("\tearly stop")
                break
            loss = AverageMeter()
            self.sfm_model.train()
            self.train_optimizer.zero_grad()

            choice = np.random.choice(train_num, self.batch_size)
            x_batch_auto = x_train_values[choice]
            y_batch_auto = y_train_values[choice]

            x_batch_auto = x_batch_auto.to(self.device)
            y_batch_auto = y_batch_auto.to(self.device)

            # forward
            preds = self.sfm_model(x_batch_auto)
            cur_loss = self.get_loss(preds, y_batch_auto, self.loss_type)
            cur_loss.backward()
            self.train_optimizer.step()
            loss.update(cur_loss.item())

            # validation
            train_loss += loss.val
            if step and step % self.eval_steps == 0:
                stop_steps += 1
                train_loss /= self.eval_steps

                with torch.no_grad():
                    self.sfm_model.eval()
                    loss_val = AverageMeter()

                    # forward
                    preds = self.sfm_model(x_val_auto)
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
                    torch.save(self.sfm_model.state_dict(), save_path)
                train_loss = 0
                # update learning rate
                self.scheduler.step(cur_loss_val)

        if self.device != 'cpu':
            torch.cuda.empty_cache()

    def get_loss(self, pred, target, loss_type):
        if loss_type == "mse":
            sqr_loss = (pred - target)**2
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
        self.sfm_model.eval()
        x_values = x_test.values
        sample_num = x_values.shape[0]
        preds = []

        for begin in range(sample_num)[::self.batch_size]:
            if sample_num-begin<self.batch_size:
                end = sample_num
            else:
                end = begin + self.batch_size

            x_batch = torch.from_numpy(x_values[begin:end]).float()

            if self.device != 'cpu':
                x_batch = x_batch.to(self.device)
            
            with torch.no_grad():
                if self.device != 'cpu':
                    pred = self.sfm_model(x_batch).detach().cpu().numpy()
                else:
                    pred = self.sfm_model(x_batch).detach().cpu().numpy()
            preds.append(pred)
        
        return pd.Series(np.concatenate(preds), index=index)

    def save(self, filename, **kwargs):
        with save_multiple_parts_file(filename) as model_dir:
            model_path = os.path.join(model_dir, os.path.split(model_dir)[-1])
            # Save model
            torch.save(self.sfm_model.state_dict(), model_path)

    def load(self, buffer, **kwargs):
        with unpack_archive_with_buffer(buffer) as model_dir:
            # Get model name
            _model_name = os.path.splitext(list(filter(lambda x: x.startswith("model.bin"), os.listdir(model_dir)))[0])[
                0
            ]
            _model_path = os.path.join(model_dir, _model_name)
            # Load model
            self.sfm_model.load_state_dict(torch.load(_model_path))
        self._fitted = True

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

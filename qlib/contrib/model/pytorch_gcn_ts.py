# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import copy

from torch.nn.functional import relu

from ...model import Model
from ...utils import get_or_create_path
from ...log import get_module_logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

from .pytorch_utils import count_parameters
from ...data.dataset.handler import DataHandlerLP
from ...contrib.model.pytorch_lstm import LSTMModel
from ...contrib.model.pytorch_gru import GRUModel


class DailyBatchSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        # calculate number of samples in each batch
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        for idx, count in zip(self.daily_index, self.daily_count):
            yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


class GCN(Model):
    """GCN Model

    Parameters
    ----------
    lr : float
        learning rate
    d_feat : int
        input dimensions for each time step
    metric : str
        the evaluation metric used in early stop
    optimizer : str
        optimizer name
    GPU : int
        the GPU ID used for training
    """

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        early_stop=20,
        loss="mse",
        base_model="GRU",
        model_path=None,
        optimizer="adam",
        GPU=0,
        n_jobs=10,
        seed=None,
        industrial_data_path="~/industry_data.csv",
        industry_col="industry_citic",
        adjacent_coef=0.01,
        **kwargs
    ):
        # Set logger.
        self.logger = get_module_logger("GCN")
        self.logger.info("GCN pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.base_model = base_model
        self.model_path = model_path
        self.n_jobs = n_jobs
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed

        self.industrial_data_path = industrial_data_path
        self.industrial = pd.read_csv(industrial_data_path, index_col=0)
        self.industry_col = industry_col
        self.adjacent_coef = adjacent_coef

        self.industry = self.industrial[self.industry_col]
        self.industry = self.industry.fillna(-1)

        self.logger.info(
            "GCN parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\nbase_model : {}"
            "\nmodel_path : {}"
            "\ndevice : {}"
            "\nuse_GPU : {}"
            "\nseed : {}"
            "\nindustry_col: {}".format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                lr,
                metric,
                early_stop,
                optimizer.lower(),
                loss,
                base_model,
                model_path,
                self.device,
                self.use_gpu,
                seed,
                industry_col,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.GCN_model = GCNModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            base_model=self.base_model,
        )
        self.logger.info("model:\n{:}".format(self.GCN_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.GCN_model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.GCN_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.GCN_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.GCN_model.to(self.device)

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

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def get_daily_inter(self, df, shuffle=False):
        # organize the train data into daily batches
        daily_count = df.groupby(level=0).size().values
        daily_index = np.roll(np.cumsum(daily_count), 1)
        daily_index[0] = 0
        if shuffle:
            # shuffle data
            daily_shuffle = list(zip(daily_index, daily_count))
            np.random.shuffle(daily_shuffle)
            daily_index, daily_count = zip(*daily_shuffle)
        return daily_index, daily_count

    def train_epoch(self, data_loader):
        self.GCN_model.train()

        for data in data_loader:
            data = data.squeeze()
            feature = data[:, :, 0:-2].to(self.device)
            label = data[:, -1, -2].to(self.device)
            ind = data[:, -1, -1]

            adjacent_matrix = (ind.reshape(-1, 1) == ind.reshape(1, -1)).float().to(self.device)

            adjacent_matrix = adjacent_matrix * self.adjacent_coef + torch.eye(adjacent_matrix.shape[0]).to(self.device)

            pred = self.GCN_model(feature.float(), adjacent_matrix)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.GCN_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        self.GCN_model.eval()

        scores = []
        losses = []

        for data in data_loader:
            data = data.squeeze()
            feature = data[:, :, 0:-2].to(self.device)
            label = data[:, -1, -2].to(self.device)
            ind = data[:, -1, -1]
            adjacent_matrix = (ind.reshape(-1, 1) == ind.reshape(1, -1)).float().to(self.device)
            adjacent_matrix = adjacent_matrix * self.adjacent_coef + torch.eye(adjacent_matrix.shape[0]).to(self.device)

            pred = self.GCN_model(feature.float(), adjacent_matrix)
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            score = self.metric_fn(pred, label)
            scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset,
        evals_result=dict(),
        save_path=None,
    ):
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        dl_train_index = dl_train.data_index.get_level_values(0)
        dl_valid_index = dl_valid.data_index.get_level_values(0)

        dl_train_ind = np.zeros(dl_train.data_arr.shape[0])
        dl_valid_ind = np.zeros(dl_valid.data_arr.shape[0])

        for col_name in dl_train.idx_df.columns:
            col_data = dl_train.idx_df[col_name]
            for val in col_data:
                if np.isnan(val) == False:
                    dl_train_ind[val] = self.industry.get(col_name, -1)
        for col_name in dl_valid.idx_df.columns:
            col_data = dl_valid.idx_df[col_name]
            for val in col_data:
                if np.isnan(val) == False:
                    dl_valid_ind[val] = self.industry.get(col_name, -1)

        dl_train.data_arr = np.concatenate([dl_train.data_arr, dl_train_ind[:, None]], axis=1)
        dl_valid.data_arr = np.concatenate([dl_valid.data_arr, dl_valid_ind[:, None]], axis=1)

        sampler_train = DailyBatchSampler(dl_train)
        sampler_valid = DailyBatchSampler(dl_valid)

        train_loader = DataLoader(dl_train, sampler=sampler_train, num_workers=self.n_jobs, drop_last=True)
        valid_loader = DataLoader(dl_valid, sampler=sampler_valid, num_workers=self.n_jobs, drop_last=True)

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # load pretrained base_model
        if self.base_model == "LSTM":
            pretrained_model = LSTMModel(d_feat=self.d_feat, hidden_size=self.hidden_size, num_layers=self.num_layers)
        elif self.base_model == "GRU":
            pretrained_model = GRUModel(d_feat=self.d_feat, hidden_size=self.hidden_size, num_layers=self.num_layers)
        else:
            raise ValueError("unknown base model name `%s`" % self.base_model)

        if self.model_path is not None:
            self.logger.info("Loading pretrained model...")
            pretrained_model.load_state_dict(torch.load(self.model_path, map_location=self.device))

        model_dict = self.GCN_model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_model.state_dict().items() if k in model_dict  # pylint: disable=E1135
        }
        model_dict.update(pretrained_dict)
        self.GCN_model.load_state_dict(model_dict)
        self.logger.info("Loading pretrained model Done...")

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

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.GCN_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.GCN_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")

        dl_test_ind = np.zeros(dl_test.data_arr.shape[0])

        for col_name in dl_test.idx_df.columns:
            col_data = dl_test.idx_df[col_name]
            for val in col_data:
                if np.isnan(val) == False:
                    dl_test_ind[val] = self.industry.get(col_name, -1)

        dl_test.data_arr = np.concatenate([dl_test.data_arr, dl_test_ind[:, None]], axis=1)

        sampler_test = DailyBatchSampler(dl_test)
        test_loader = DataLoader(dl_test, sampler=sampler_test, num_workers=self.n_jobs)
        self.GCN_model.eval()
        preds = []

        for data in test_loader:
            data = data.squeeze()
            feature = data[:, :, 0:-2].to(self.device)
            ind = data[:, -1, -1].to(self.device)
            adjacent_matrix = (ind.reshape(-1, 1) == ind.reshape(1, -1)).float().to(self.device)
            adjacent_matrix = adjacent_matrix * self.adjacent_coef + torch.eye(adjacent_matrix.shape[0]).to(self.device)

            with torch.no_grad():
                pred = self.GCN_model(feature.float(), adjacent_matrix).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=relu):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation

    def forward(self, x, adj):
        # x: [N, in_dim]
        x = torch.matmul(adj, x)
        # x: [N, ind_im]
        x = self.linear(x)
        # x: [N, out_dim]
        if self.activation is not None:
            x = self.activation(x)
        return x


class GCNModel(nn.Module):
    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        base_model="GRU",
        base_model_trainable=True,
        gcn_num_layers=2,
        adj_coef=0.01,
    ):
        super().__init__()

        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)

        if not base_model_trainable:
            for param in self.rnn.parameters():
                param.requires_grad = False

        self.hidden_size = hidden_size
        self.d_feat = d_feat
        self.fc_out = nn.Linear(hidden_size, 1)
        self.adj_coef = adj_coef

        self.GCN = torch.nn.ModuleList(
            [GCNLayer(in_dim=hidden_size, out_dim=hidden_size, activation=relu) for _ in range(gcn_num_layers)]
        )

    def forward(self, x, adj):
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]

        adj = adj / torch.sqrt(adj.sum(axis=1, keepdim=True)) / torch.sqrt(adj.sum(axis=0, keepdim=True))
        for gcn in self.GCN:
            hidden = gcn(hidden, adj)

        return self.fc_out(hidden).squeeze()
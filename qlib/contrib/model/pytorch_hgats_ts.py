# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import copy
from ...utils import get_or_create_path
from ...log import get_module_logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch import pinverse as pinv

from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...contrib.model.pytorch_lstm import LSTMModel
from ...contrib.model.pytorch_gru import GRUModel


class IdxSampler:
    def __init__(self, sampler):
        self.sampler = sampler

    def __getitem__(self, i: int):
        return self.sampler[i], i

    def __len__(self):
        return len(self.sampler)


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


class HGATs(Model):
    """HGATs Model

    Parameters
    ----------
    lr : float
        learning rate
    d_feat : int
        input dimensions for each time step
    metric : str
        the evaluate metric used in early stop
    optimizer : str
        optimizer name
    GPU : int
        the GPU ID used for training
    """

    def __init__(
        self,
        d_feat=20,
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
        GPU="0",
        n_jobs=10,
        seed=None,
        **kwargs
    ):
        # Set logger.
        self.logger = get_module_logger("HGATs")
        self.logger.info("HGATs pytorch version...")

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
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.n_jobs = n_jobs
        self.seed = seed

        self.logger.info(
            "HGATs parameters setting:"
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
            "\nvisible_GPU : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
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
                GPU,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.HGAT_model = HGATModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            base_model=self.base_model,
        )
        self.logger.info("model:\n{:}".format(self.HGAT_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.HGAT_model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.HGAT_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.HGAT_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.HGAT_model.to(self.device)

        self.bigG = pd.read_hdf("benchmarks/HGATs/hypergraph/CSI300.h5", "df")
        self.bigG = self.bigG.swaplevel().sort_index()  # (date, stock, industry)

        self.num_ind = 29 + 1  # number of industries plus an 'unknown' one

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

        if self.metric == "" or self.metric == "loss":
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, data_loader, tsds, big_G):

        self.HGAT_model.train()

        for data, i in data_loader:

            data = data.squeeze()
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            date = tsds.get_index()[i.tolist()].to_frame()["datetime"][0]
            stockbatch = tsds.get_index()[i.tolist()].to_frame()["instrument"].values

            G = big_G.loc[date]  # (stock, industry)
            G = G[G.index.isin(stockbatch)]
            unknown_stocks = stockbatch[~np.isin(stockbatch, G.index.values)]
            # assign 'nan industry' labels to the unknown stocks
            for newstock in unknown_stocks:
                G.loc[newstock] = np.nan

            GH = pd.DataFrame(
                0.0,
                index=np.arange(len(G)),
                columns=np.append(np.arange(1, self.num_ind), np.nan),
            )

            GH.update(pd.get_dummies(G.values.squeeze()).astype("float64"))  # [#stocks, #industries]

            pred = self.HGAT_model(feature.float(), torch.Tensor(GH.values).to(self.device))
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.HGAT_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader, tsds, big_G):

        self.HGAT_model.eval()

        scores = []
        losses = []

        for data, i in data_loader:

            data = data.squeeze()
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            date = tsds.get_index()[i.tolist()].to_frame()["datetime"][0]
            stockbatch = tsds.get_index()[i.tolist()].to_frame()["instrument"].values

            G = big_G.loc[date]  # (stock, industry)
            G = G[G.index.isin(stockbatch)]
            unknown_stocks = stockbatch[~np.isin(stockbatch, G.index.values)]
            # assign 'nan industry' labels to the unknown stocks
            for newstock in unknown_stocks:
                G.loc[newstock] = np.nan

            GH = pd.DataFrame(
                0.0,
                index=np.arange(len(G)),
                columns=np.append(np.arange(1, self.num_ind), np.nan),
            )

            GH.update(pd.get_dummies(G.values.squeeze()).astype("float64"))  # [#stocks, #industries]

            pred = self.HGAT_model(feature.float(), torch.Tensor(GH.values).to(self.device))
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

        sampler_train = DailyBatchSampler(dl_train)
        sampler_valid = DailyBatchSampler(dl_valid)

        train_loader = DataLoader(
            IdxSampler(dl_train),
            sampler=sampler_train,
            num_workers=self.n_jobs,
            drop_last=True,
        )
        valid_loader = DataLoader(
            IdxSampler(dl_valid),
            sampler=sampler_valid,
            num_workers=self.n_jobs,
            drop_last=True,
        )

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # load pretrained base_model
        if self.base_model == "LSTM":
            pretrained_model = LSTMModel(
                d_feat=self.d_feat,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
            )
        elif self.base_model == "GRU":
            pretrained_model = GRUModel(
                d_feat=self.d_feat,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
            )
        else:
            raise ValueError("unknown base model name `%s`" % self.base_model)

        if self.model_path is not None:
            self.logger.info("Loading pretrained model...")
            pretrained_model.load_state_dict(torch.load(self.model_path))

        model_dict = self.HGAT_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.HGAT_model.load_state_dict(model_dict)
        self.logger.info("Loading pretrained model Done...")

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader, dl_train, self.bigG)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader, dl_train, self.bigG)
            val_loss, val_score = self.test_epoch(valid_loader, dl_valid, self.bigG)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.HGAT_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.HGAT_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        sampler_test = DailyBatchSampler(dl_test)
        test_loader = DataLoader(IdxSampler(dl_test), sampler=sampler_test, num_workers=self.n_jobs)
        self.HGAT_model.eval()
        preds = []

        for data, i in test_loader:

            data = data.squeeze()
            feature = data[:, :, 0:-1].to(self.device)

            date = dl_test.get_index()[i.tolist()].to_frame()["datetime"][0]
            stockbatch = dl_test.get_index()[i.tolist()].to_frame()["instrument"].values

            G = self.bigG.loc[date]  # (stock, industry)
            G = G[G.index.isin(stockbatch)]
            unknown_stocks = stockbatch[~np.isin(stockbatch, G.index.values)]
            # assign 'nan industry' labels to the unknown stocks
            for newstock in unknown_stocks:
                G.loc[newstock] = np.nan

            GH = pd.DataFrame(
                0.0,
                index=np.arange(len(G)),
                columns=np.append(np.arange(1, self.num_ind), np.nan),
            )

            GH.update(pd.get_dummies(G.values.squeeze()).astype("float64"))  # [#stocks, #industries]

            with torch.no_grad():
                pred = self.HGAT_model(feature.float(), torch.Tensor(GH.values).to(self.device)).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(
            np.concatenate(preds),
            index=dl_test.get_index()[: len(np.concatenate(preds))],
        )


class HGATModel(nn.Module):
    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        num_ind=29 + 1,  # number of industries plus an 'unknown' one
        base_model="GRU",
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

        self.hidden_size = hidden_size
        self.num_ind = num_ind
        self.d_feat = d_feat
        self.transformation = nn.Linear(self.hidden_size, self.hidden_size)
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
        self.a.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def cal_attention(self, x, y):
        x = self.transformation(x)  # x <- hidden
        y = self.transformation(y)  # y <- hidden_agg

        sample_num = x.shape[0]  # num of stocks
        edge_num = y.shape[0]  # num of industries
        dim = x.shape[1]  # num of temporal features
        e_x = x.expand(edge_num, sample_num, dim)
        e_y = y.expand(sample_num, edge_num, dim)
        e_y = torch.transpose(e_y, 0, 1)
        attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)  # P i || P j
        self.a_t = torch.t(self.a)
        attention_out = self.a_t.mm(torch.t(attention_in)).view(edge_num, sample_num).t()
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)

        return att_weight

    def forward(self, x, GH):

        out, _ = self.rnn(x)
        hidden = out[:, -1, :]  # [#stocks, #features]

        hidden_agg = torch.t(GH).mm(hidden)  # [#industries, #features]

        att_weight = self.cal_attention(hidden, hidden_agg)  # [#stocks, #industries]

        De = pinv(torch.diag(GH.sum(axis=0)))
        Dv = pinv(torch.diag(GH.sum(axis=1) ** 1 / 2))
        H = (Dv).mm(att_weight).mm(De).mm(att_weight.T).mm(Dv)

        hidden = H.mm(hidden) + hidden
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)
        return self.fc_out(hidden).squeeze()


"""
Be aware:
- The stock-industry hypergraph is different from day to day due to missing data, changes to the CSI components, and changes to the industry labels of certain stocks.
- When we encounter stocks with unknown industry information, we them with 'nan industry' labels.
"""

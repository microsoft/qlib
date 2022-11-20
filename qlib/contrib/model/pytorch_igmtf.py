# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from typing import Text, Union
import copy
from ...utils import get_or_create_path
from ...log import get_module_logger

import torch
import torch.nn as nn
import torch.optim as optim

from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...contrib.model.pytorch_lstm import LSTMModel
from ...contrib.model.pytorch_gru import GRUModel


class IGMTF(Model):
    """IGMTF Model

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
        seed=None,
        **kwargs
    ):
        # Set logger.
        self.logger = get_module_logger("IGMTF")
        self.logger.info("IMGTF pytorch version...")

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
        self.seed = seed

        self.logger.info(
            "IGMTF parameters setting:"
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

        self.igmtf_model = IGMTFModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            base_model=self.base_model,
        )
        self.logger.info("model:\n{:}".format(self.igmtf_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.igmtf_model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.igmtf_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.igmtf_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.igmtf_model.to(self.device)

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

        if self.metric == "ic":
            x = pred[mask]
            y = label[mask]

            vx = x - torch.mean(x)
            vy = y - torch.mean(y)
            return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)))

        if self.metric == ("", "loss"):
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

    def get_train_hidden(self, x_train):
        x_train_values = x_train.values
        daily_index, daily_count = self.get_daily_inter(x_train, shuffle=True)
        self.igmtf_model.eval()
        train_hidden = []
        train_hidden_day = []

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_train_values[batch]).float().to(self.device)
            out = self.igmtf_model(feature, get_hidden=True)
            train_hidden.append(out.detach().cpu())
            train_hidden_day.append(out.detach().cpu().mean(dim=0).unsqueeze(dim=0))

        train_hidden = np.asarray(train_hidden, dtype=object)
        train_hidden_day = torch.cat(train_hidden_day)

        return train_hidden, train_hidden_day

    def train_epoch(self, x_train, y_train, train_hidden, train_hidden_day):

        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)

        self.igmtf_model.train()

        daily_index, daily_count = self.get_daily_inter(x_train, shuffle=True)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_train_values[batch]).float().to(self.device)
            label = torch.from_numpy(y_train_values[batch]).float().to(self.device)
            pred = self.igmtf_model(feature, train_hidden=train_hidden, train_hidden_day=train_hidden_day)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.igmtf_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y, train_hidden, train_hidden_day):

        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.igmtf_model.eval()

        scores = []
        losses = []

        daily_index, daily_count = self.get_daily_inter(data_x, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)
            label = torch.from_numpy(y_values[batch]).float().to(self.device)

            pred = self.igmtf_model(feature, train_hidden=train_hidden, train_hidden_day=train_hidden_day)
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            score = self.metric_fn(pred, label)
            scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):

        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # load pretrained base_model
        if self.base_model == "LSTM":
            pretrained_model = LSTMModel()
        elif self.base_model == "GRU":
            pretrained_model = GRUModel()
        else:
            raise ValueError("unknown base model name `%s`" % self.base_model)

        if self.model_path is not None:
            self.logger.info("Loading pretrained model...")
            pretrained_model.load_state_dict(torch.load(self.model_path, map_location=self.device))

        model_dict = self.igmtf_model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_model.state_dict().items() if k in model_dict  # pylint: disable=E1135
        }
        model_dict.update(pretrained_dict)
        self.igmtf_model.load_state_dict(model_dict)
        self.logger.info("Loading pretrained model Done...")

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            train_hidden, train_hidden_day = self.get_train_hidden(x_train)
            self.train_epoch(x_train, y_train, train_hidden, train_hidden_day)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(x_train, y_train, train_hidden, train_hidden_day)
            val_loss, val_score = self.test_epoch(x_valid, y_valid, train_hidden, train_hidden_day)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.igmtf_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.igmtf_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        x_train = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
        train_hidden, train_hidden_day = self.get_train_hidden(x_train)
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.igmtf_model.eval()
        x_values = x_test.values
        preds = []

        daily_index, daily_count = self.get_daily_inter(x_test, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            x_batch = torch.from_numpy(x_values[batch]).float().to(self.device)

            with torch.no_grad():
                pred = (
                    self.igmtf_model(x_batch, train_hidden=train_hidden, train_hidden_day=train_hidden_day)
                    .detach()
                    .cpu()
                    .numpy()
                )

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


class IGMTFModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
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
        self.lins = nn.Sequential()
        for i in range(2):
            self.lins.add_module("linear" + str(i), nn.Linear(hidden_size, hidden_size))
            self.lins.add_module("leakyrelu" + str(i), nn.LeakyReLU())
        self.fc_output = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.project1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.project2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_out_pred = nn.Linear(hidden_size * 2, 1)

        self.leaky_relu = nn.LeakyReLU()
        self.d_feat = d_feat

    def cal_cos_similarity(self, x, y):  # the 2nd dimension of x and y are the same
        xy = x.mm(torch.t(y))
        x_norm = torch.sqrt(torch.sum(x * x, dim=1)).reshape(-1, 1)
        y_norm = torch.sqrt(torch.sum(y * y, dim=1)).reshape(-1, 1)
        cos_similarity = xy / (x_norm.mm(torch.t(y_norm)) + 1e-6)
        return cos_similarity

    def sparse_dense_mul(self, s, d):
        i = s._indices()
        v = s._values()
        dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
        return torch.sparse.FloatTensor(i, v * dv, s.size())

    def forward(self, x, get_hidden=False, train_hidden=None, train_hidden_day=None, k_day=10, n_neighbor=10):
        # x: [N, F*T]
        device = x.device
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.lins(out)
        mini_batch_out = out
        if get_hidden is True:
            return mini_batch_out

        mini_batch_out_day = torch.mean(mini_batch_out, dim=0).unsqueeze(0)
        day_similarity = self.cal_cos_similarity(mini_batch_out_day, train_hidden_day.to(device))
        day_index = torch.topk(day_similarity, k_day, dim=1)[1]
        sample_train_hidden = train_hidden[day_index.long().cpu()].squeeze()
        sample_train_hidden = torch.cat(list(sample_train_hidden)).to(device)
        sample_train_hidden = self.lins(sample_train_hidden)
        cos_similarity = self.cal_cos_similarity(self.project1(mini_batch_out), self.project2(sample_train_hidden))

        row = (
            torch.linspace(0, x.shape[0] - 1, x.shape[0])
            .reshape([-1, 1])
            .repeat(1, n_neighbor)
            .reshape(1, -1)
            .to(device)
        )
        column = torch.topk(cos_similarity, n_neighbor, dim=1)[1].reshape(1, -1)
        mask = torch.sparse_coo_tensor(
            torch.cat([row, column]),
            torch.ones([row.shape[1]]).to(device) / n_neighbor,
            (x.shape[0], sample_train_hidden.shape[0]),
        )
        cos_similarity = self.sparse_dense_mul(mask, cos_similarity)

        agg_out = torch.sparse.mm(cos_similarity, self.project2(sample_train_hidden))
        # out = self.fc_out(out).squeeze()
        out = self.fc_out_pred(torch.cat([mini_batch_out, agg_out], axis=1)).squeeze()
        return out

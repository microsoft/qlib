# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from typing import Text, Union
import copy

from torch.nn.functional import relu

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
        seed=None,
        industrial_data_path="~/industry_data.csv",
        industry_col="industry",
        smooth_perplexity=1,
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
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed

        self.industrial_data_path = industrial_data_path
        self.industrial = pd.read_csv(industrial_data_path, index_col=0)
        self.industry_col = industry_col
        self.adjacent_coef = adjacent_coef

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
                self.device,
                self.use_gpu,
                seed,
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

    def get_industry(self, date, instruments):
        table = self.industrial.loc[:, self.industry_col]

        instrument_to_industry = {}

        for id, instrument in enumerate(instruments):
            industry = table.get(instrument, -1)
            if np.isnan(industry):
                industry = -1
            instrument_to_industry[instrument] = industry
        return instrument_to_industry

    def adjacent_matrix(self, groups):
        # extract the adjacent matrix
        industries = []
        for i, (idx, count) in enumerate(groups):
            for j in range(count):
                industries.append(i)
        industries = np.array(industries)
        adjacent_matrix = (industries.reshape(-1, 1) == industries.reshape(1, -1)).astype(np.float32)
        return adjacent_matrix

    def get_input_data(self, data_x, data_y, data_x_values, data_y_values, idx, count):
        instruments = []
        index = []
        for i in range(idx, idx + count):
            index.append(i)
            instruments.append(data_x.index[i][1])

        instrument_to_idx = {instrument: idx for idx, instrument in zip(index, instruments)}

        date = str(data_x.index[idx][0])[:10]
        instrument_to_industry = self.get_industry(date, instruments)
        group_by_industry = list(instrument_to_industry.items())
        group_by_industry.sort(key=lambda x: x[1])
        # for each industry, get the starting index and the number of instruments
        groups = []
        now_industry = None
        index = []
        for id, (instrument, industry) in enumerate(group_by_industry):
            if industry != now_industry:
                now_industry = industry
                groups.append((id, 1))
            else:
                groups[-1] = (groups[-1][0], groups[-1][1] + 1)
            index.append(instrument_to_idx[instrument])
        index = np.array(index)

        label = data_y_values[index] if data_y_values is not None else None

        A = torch.tensor(self.adjacent_matrix(groups)) * self.adjacent_coef

        return data_x_values[index], label, (A + torch.eye(A.shape[0])).to(self.device), index

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)
        self.GCN_model.train()

        # organize the train data into daily batches
        daily_index, daily_count = self.get_daily_inter(x_train, shuffle=True)

        for idx, count in zip(daily_index, daily_count):
            feature, label, groups, _ = self.get_input_data(
                x_train, y_train, x_train_values, y_train_values, idx, count
            )

            feature = torch.from_numpy(feature).float().to(self.device)
            label = torch.from_numpy(label).float().to(self.device)

            pred = self.GCN_model(feature, groups)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.GCN_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.GCN_model.eval()

        scores = []
        losses = []

        # organize the test data into daily batches
        daily_index, daily_count = self.get_daily_inter(data_x, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            feature, label, groups, _ = self.get_input_data(data_x, data_y, x_values, y_values, idx, count)

            feature = torch.from_numpy(feature).float().to(self.device)
            label = torch.from_numpy(label).float().to(self.device)

            pred = self.GCN_model(feature, groups)
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
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        save_path = get_or_create_path(save_path)
        stop_steps = 0
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
            self.train_epoch(x_train, y_train)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(x_train, y_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
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

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(segment, col_set="feature")
        final_index = x_test.index
        self.GCN_model.eval()
        x_values = x_test.values
        preds = []

        # organize the data into daily batches
        daily_index, daily_count = self.get_daily_inter(x_test, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            feature, label, groups, index = self.get_input_data(x_test, None, x_values, None, idx, count)

            feature = torch.from_numpy(feature).float().to(self.device)

            with torch.no_grad():
                pred = self.GCN_model(feature, groups).detach().cpu().numpy()

                decoder = [0] * count
                for i, id in enumerate(index):
                    decoder[id - idx] = i
                pred = pred[decoder]
            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=final_index)


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
        # x: [N, F*T]
        x = x.reshape(x.shape[0], self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]

        adj = adj / torch.sqrt(adj.sum(axis=1, keepdim=True)) / torch.sqrt(adj.sum(axis=0, keepdim=True))
        for gcn in self.GCN:
            hidden = gcn(hidden, adj)
        # hidden = self.GCN_3(hidden, adj)

        return self.fc_out(hidden).squeeze()


if __name__ == "__main__":
    pass

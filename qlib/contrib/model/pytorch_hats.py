# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import copy
from ...utils import create_save_path
from ...log import get_module_logger

import torch
import torch.nn as nn
import torch.optim as optim

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP


class HATS(Model):
    """HATS Model

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
        dropout=0.5,
        n_epochs=200,
        lr=0.0001,
        metric="loss",
        early_stop=20,
        loss="mse",
        base_model="LSTM",
        with_pretrain=True,
        optimizer="adam",
        GPU="0",
        seed=0,
        **kwargs
    ):
        # Set logger.
        self.logger = get_module_logger("HATS")
        self.logger.info("HATS pytorch version...")

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
        self.with_pretrain = with_pretrain
        self.visible_GPU = GPU
        self.use_gpu = torch.cuda.is_available()
        self.seed = seed

        self.logger.info(
            "HATS parameters setting:"
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
            "\nwith_pretrain : {}"
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
                with_pretrain,
                GPU,
                self.use_gpu,
                seed,
            )
        )

        self.HATS_model = HATSModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            base_model=self.base_model,
        )
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.HATS_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.HATS_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self._fitted = False
        if self.use_gpu:
            self.HATS_model.cuda()
            # set the visible GPU
            if self.visible_GPU:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.visible_GPU

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
        if self.metric == "IC":
            return self.cal_ic(pred[mask], label[mask])

        if self.metric == "" or self.metric == "loss":  # use loss
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def cal_ic(self, pred, label):
        return torch.mean(pred * label)

    def get_daily_inter(self, df, shuffle=False):
        # organize the train data into daily inter as daily batches
        daily_count = df.groupby(level=0).size().values
        daily_index = np.roll(np.cumsum(daily_count), 1)
        daily_index[0] = 0
        if shuffle:
            # shuffle the daily inter data
            daily_shuffle = list(zip(daily_index, daily_count))
            np.random.shuffle(daily_shuffle)
            daily_index, daily_count = zip(*daily_shuffle)
        return daily_index, daily_count

    def train_epoch(self, x_train, y_train):

        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)

        self.HATS_model.train()

        # organize the train data into daily inter as daily batches
        daily_index, daily_count = self.get_daily_inter(x_train, shuffle=True)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_train_values[batch]).float()
            label = torch.from_numpy(y_train_values[batch]).float()

            if self.use_gpu:
                feature = feature.cuda()
                label = label.cuda()

            pred = self.HATS_model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.HATS_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):

        # prepare testing data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.HATS_model.eval()

        scores = []
        losses = []

        # organize the test data into daily inter as daily batches
        daily_index, daily_count = self.get_daily_inter(data_x, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float()
            label = torch.from_numpy(y_values[batch]).float()

            if self.use_gpu:
                feature = feature.cuda()
                label = label.cuda()

            pred = self.HATS_model(feature)
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            score = self.metric_fn(pred, label)
            scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        verbose=True,
        save_path=None,
    ):

        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        if save_path == None:
            save_path = create_save_path(save_path)
        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # load pretrained base_model
        if self.with_pretrain:
            self.logger.info("Loading pretrained model...")
            if self.base_model == "LSTM":
                from ...contrib.model.pytorch_lstm import LSTMModel

                pretrained_model = LSTMModel()
                pretrained_model.load_state_dict(torch.load("benchmarks/LSTM/model_lstm_csi300.pkl"))
            elif self.base_model == "GRU":
                from ...contrib.model.pytorch_gru import GRUModel

                pretrained_model = GRUModel()
                pretrained_model.load_state_dict(torch.load("benchmarks/GRU/model_gru_csi300.pkl"))
            model_dict = self.HATS_model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.HATS_model.load_state_dict(model_dict)
            self.logger.info("Loading pretrained model Done...")

        # train
        self.logger.info("training...")
        self._fitted = True

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
                best_param = copy.deepcopy(self.HATS_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.HATS_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self._fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare("test", col_set="feature")
        index = x_test.index
        self.HATS_model.eval()
        x_values = x_test.values
        sample_num = x_values.shape[0]
        preds = []

        # organize the data into daily inter as daily batches
        daily_index, daily_count = self.get_daily_inter(x_test, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            x_batch = torch.from_numpy(x_values[batch]).float()

            if self.use_gpu:
                x_batch = x_batch.cuda()

            with torch.no_grad():
                if self.use_gpu:
                    pred = self.HATS_model(x_batch).detach().cpu().numpy()
                else:
                    pred = self.HATS_model(x_batch).detach().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


class HATSModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()

        if base_model == "GRU":
            self.model = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.model = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)

        self.hidden_size = hidden_size
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size, track_running_stats=False)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_size, track_running_stats=False)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.d_feat = d_feat

        num_head_att = [1] * num_layers
        hidden_dim = [hidden_size] * num_layers
        dims = [d_feat] + [d * nh for (d, nh) in zip(hidden_dim, num_head_att[:-1])] + [num_head_att[-1]]
        in_dims = dims[:-1]
        out_dims = [d // nh for (d, nh) in zip(dims[1:], num_head_att)]
        self.attn = nn.ModuleList(
            [GraphAttention(i, o, nh, dropout) for (i, o, nh) in zip(in_dims, out_dims, num_head_att)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(dim) for dim in dims[1:-1]])
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()

    def forward(self, x):
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.model(x)
        hidden = out[:, -1, :]
        hidden = self.bn1(hidden)
        attention = GraphAttention.cal_attention(hidden, hidden)
        output = attention.mm(hidden)
        output = self.fc(output)
        output = self.bn2(output)
        output = self.leaky_relu(output)
        return self.fc_out(output).squeeze()


class GraphAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, dropout=0.5):

        super().__init__()

        """
        Parameters
        ----------
        input_dim : int
            Dimension of input node features.
        output_dim : int
            Dimension of output node features.
        num_heads : list of ints
            Number of attention heads in each hidden layer and output layer. Must be non empty. Note that len(num_heads) = len(hidden_dims)+1.
        dropout : float
            Dropout rate. Default: 0.5.
        """

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        self.fcs = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_heads)])
        self.a = nn.ModuleList([nn.Linear(2 * output_dim, 1) for _ in range(num_heads)])

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, features, nodes, mappings, rows):

        """
        Parameters
        ----------
        features : torch.Tensor
            An (n' x input_dim) tensor of input node features.
        nodes : list of numpy array
            nodes[i] is an array of the nodes in the ith layer of the
            computation graph.
        mappings : list of dictionary
            mappings[i] is a dictionary mappings node v (labelled 0 to |V|-1)
            in nodes[i] to its position in nodes[i]. For example,
            if nodes[i] = [2,5], then mappings[i][2] = 0 and
            mappings[i][5] = 1.
        rows : numpy array
            rows[i] is an array of neighbors of node i.
        Returns
        -------
        out : torch.Tensor
            An (len(node_layers[-1]) x output_dim) tensor of output node features.
        """

        nprime = features.shape[0]
        rows = [np.array([mappings[v] for v in row], dtype=np.int64) for row in rows]
        sum_degs = np.hstack(([0], np.cumsum([len(row) for row in rows])))
        mapped_nodes = [mappings[v] for v in nodes]
        indices = torch.LongTensor([[v, c] for (v, row) in zip(mapped_nodes, rows) for c in row]).t()

        out = []
        for k in range(self.num_heads):
            h = self.fcs[k](features)

            nbr_h = torch.cat(tuple([h[row] for row in rows]), dim=0)
            self_h = torch.cat(
                tuple([h[mappings[nodes[i]]].repeat(len(row), 1) for (i, row) in enumerate(rows)]), dim=0
            )
            cat_h = torch.cat((self_h, nbr_h), dim=1)

            e = self.leakyrelu(self.a[k](cat_h))

            alpha = [self.softmax(e[lo:hi]) for (lo, hi) in zip(sum_degs, sum_degs[1:])]
            alpha = torch.cat(tuple(alpha), dim=0)
            alpha = alpha.squeeze(1)
            alpha = self.dropout(alpha)

            adj = torch.sparse.FloatTensor(indices, alpha, torch.Size([nprime, nprime]))
            out.append(torch.sparse.mm(adj, h)[mapped_nodes])

        return out

    @staticmethod
    def cal_attention(x, y):
        att_x = torch.mean(x, dim=1).reshape(-1, 1)
        att_y = torch.mean(y, dim=1).reshape(-1, 1)
        att = att_x.mm(torch.t(att_y))
        return (
            torch.mean(
                x.reshape(x.shape[0], 1, x.shape[1]).repeat(1, y.shape[0], 1)
                * y.reshape(1, y.shape[0], y.shape[1]).repeat(x.shape[0], 1, 1),
                dim=2,
            )
            - att
        )

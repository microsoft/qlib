# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import division
from __future__ import print_function


import copy
import math
from typing import Text, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from qlib.contrib.model.pytorch_gru import GRUModel
from qlib.contrib.model.pytorch_lstm import LSTMModel
from qlib.contrib.model.pytorch_utils import count_parameters
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import CSRankNorm
from qlib.log import get_module_logger
from qlib.model.base import Model
from qlib.utils import get_or_create_path
from torch.autograd import Function


class ADD(Model):
    """ADD Model

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
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        dec_dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="mse",
        batch_size=5000,
        early_stop=20,
        base_model="GRU",
        model_path=None,
        optimizer="adam",
        gamma=0.1,
        gamma_clip=0.4,
        mu=0.05,
        GPU=0,
        seed=None,
        **kwargs
    ):
        # Set logger.
        self.logger = get_module_logger("ADD")
        self.logger.info("ADD pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.dec_dropout = dec_dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.base_model = base_model
        self.model_path = model_path
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed

        self.gamma = gamma
        self.gamma_clip = gamma_clip
        self.mu = mu

        self.logger.info(
            "ADD parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\ndec_dropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nbase_model : {}"
            "\nmodel_path : {}"
            "\ngamma : {}"
            "\ngamma_clip : {}"
            "\nmu : {}"
            "\ndevice : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                dec_dropout,
                n_epochs,
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                base_model,
                model_path,
                gamma,
                gamma_clip,
                mu,
                self.device,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.ADD_model = ADDModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            dec_dropout=self.dec_dropout,
            base_model=self.base_model,
            gamma=self.gamma,
            gamma_clip=self.gamma_clip,
        )
        self.logger.info("model:\n{:}".format(self.ADD_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.ADD_model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.ADD_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.ADD_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.ADD_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def loss_pre_excess(self, pred_excess, label_excess, record=None):
        mask = ~torch.isnan(label_excess)
        pre_excess_loss = F.mse_loss(pred_excess[mask], label_excess[mask])
        if record is not None:
            record["pre_excess_loss"] = pre_excess_loss.item()
        return pre_excess_loss

    def loss_pre_market(self, pred_market, label_market, record=None):
        pre_market_loss = F.cross_entropy(pred_market, label_market)
        if record is not None:
            record["pre_market_loss"] = pre_market_loss.item()
        return pre_market_loss

    def loss_pre(self, pred_excess, label_excess, pred_market, label_market, record=None):
        pre_loss = self.loss_pre_excess(pred_excess, label_excess, record) + self.loss_pre_market(
            pred_market, label_market, record
        )
        if record is not None:
            record["pre_loss"] = pre_loss.item()
        return pre_loss

    def loss_adv_excess(self, adv_excess, label_excess, record=None):
        mask = ~torch.isnan(label_excess)
        adv_excess_loss = F.mse_loss(adv_excess.squeeze()[mask], label_excess[mask])
        if record is not None:
            record["adv_excess_loss"] = adv_excess_loss.item()
        return adv_excess_loss

    def loss_adv_market(self, adv_market, label_market, record=None):
        adv_market_loss = F.cross_entropy(adv_market, label_market)
        if record is not None:
            record["adv_market_loss"] = adv_market_loss.item()
        return adv_market_loss

    def loss_adv(self, adv_excess, label_excess, adv_market, label_market, record=None):
        adv_loss = self.loss_adv_excess(adv_excess, label_excess, record) + self.loss_adv_market(
            adv_market, label_market, record
        )
        if record is not None:
            record["adv_loss"] = adv_loss.item()
        return adv_loss

    def loss_fn(self, x, preds, label_excess, label_market, record=None):
        loss = (
            self.loss_pre(preds["excess"], label_excess, preds["market"], label_market, record)
            + self.loss_adv(preds["adv_excess"], label_excess, preds["adv_market"], label_market, record)
            + self.mu * self.loss_rec(x, preds["reconstructed_feature"], record)
        )
        if record is not None:
            record["loss"] = loss.item()
        return loss

    def loss_rec(self, x, rec_x, record=None):
        x = x.reshape(len(x), self.d_feat, -1)
        x = x.permute(0, 2, 1)
        rec_loss = F.mse_loss(x, rec_x)
        if record is not None:
            record["rec_loss"] = rec_loss.item()
        return rec_loss

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

    def cal_ic_metrics(self, pred, label):
        metrics = {}
        metrics["mse"] = -F.mse_loss(pred, label).item()
        metrics["loss"] = metrics["mse"]
        pred = pd.Series(pred.cpu().detach().numpy())
        label = pd.Series(label.cpu().detach().numpy())
        metrics["ic"] = pred.corr(label)
        metrics["ric"] = pred.corr(label, method="spearman")
        return metrics

    def test_epoch(self, data_x, data_y, data_m):
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)
        m_values = np.squeeze(data_m.values.astype(int))
        self.ADD_model.eval()

        metrics_list = []

        daily_index, daily_count = self.get_daily_inter(data_x, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)
            label_excess = torch.from_numpy(y_values[batch]).float().to(self.device)
            label_market = torch.from_numpy(m_values[batch]).long().to(self.device)

            metrics = {}
            preds = self.ADD_model(feature)
            self.loss_fn(feature, preds, label_excess, label_market, metrics)
            metrics.update(self.cal_ic_metrics(preds["excess"], label_excess))
            metrics_list.append(metrics)
        metrics = {}
        keys = metrics_list[0].keys()
        for k in keys:
            vs = [m[k] for m in metrics_list]
            metrics[k] = sum(vs) / len(vs)

        return metrics

    def train_epoch(self, x_train_values, y_train_values, m_train_values):
        self.ADD_model.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        cur_step = 1

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break
            batch = indices[i : i + self.batch_size]
            feature = torch.from_numpy(x_train_values[batch]).float().to(self.device)
            label_excess = torch.from_numpy(y_train_values[batch]).float().to(self.device)
            label_market = torch.from_numpy(m_train_values[batch]).long().to(self.device)

            preds = self.ADD_model(feature)

            loss = self.loss_fn(feature, preds, label_excess, label_market)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.ADD_model.parameters(), 3.0)
            self.train_optimizer.step()
            cur_step += 1

    def log_metrics(self, mode, metrics):
        metrics = ["{}/{}: {:.6f}".format(k, mode, v) for k, v in metrics.items()]
        metrics = ", ".join(metrics)
        self.logger.info(metrics)

    def bootstrap_fit(self, x_train, y_train, m_train, x_valid, y_valid, m_valid):
        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0

        # train
        self.logger.info("training...")
        self.fitted = True
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)
        m_train_values = np.squeeze(m_train.values.astype(int))

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(x_train_values, y_train_values, m_train_values)
            self.logger.info("evaluating...")
            train_metrics = self.test_epoch(x_train, y_train, m_train)
            valid_metrics = self.test_epoch(x_valid, y_valid, m_valid)
            self.log_metrics("train", train_metrics)
            self.log_metrics("valid", valid_metrics)

            if self.metric in valid_metrics:
                val_score = valid_metrics[self.metric]
            else:
                raise ValueError("unknown metric name `%s`" % self.metric)
            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.ADD_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break
            self.ADD_model.before_adv_excess.step_alpha()
            self.ADD_model.before_adv_market.step_alpha()
        self.logger.info("bootstrap_fit best score: {:.6f} @ {}".format(best_score, best_epoch))
        self.ADD_model.load_state_dict(best_param)
        return best_score

    def gen_market_label(self, df, raw_label):
        market_label = raw_label.groupby("datetime").mean().squeeze()
        bins = [-np.inf, self.lo, self.hi, np.inf]
        market_label = pd.cut(market_label, bins, labels=False)
        market_label.name = ("market_return", "market_return")
        df = df.join(market_label)
        return df

    def fit_thresh(self, train_label):
        market_label = train_label.groupby("datetime").mean().squeeze()
        self.lo, self.hi = market_label.quantile([1 / 3, 2 / 3])

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):
        label_train, label_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["label"],
            data_key=DataHandlerLP.DK_R,
        )
        self.fit_thresh(label_train)
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        df_train = self.gen_market_label(df_train, label_train)
        df_valid = self.gen_market_label(df_valid, label_valid)

        x_train, y_train, m_train = df_train["feature"], df_train["label"], df_train["market_return"]
        x_valid, y_valid, m_valid = df_valid["feature"], df_valid["label"], df_valid["market_return"]

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

            model_dict = self.ADD_model.enc_excess.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_model.rnn.state_dict().items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.ADD_model.enc_excess.load_state_dict(model_dict)
            model_dict = self.ADD_model.enc_market.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_model.rnn.state_dict().items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.ADD_model.enc_market.load_state_dict(model_dict)
            self.logger.info("Loading pretrained model Done...")

        self.bootstrap_fit(x_train, y_train, m_train, x_valid, y_valid, m_valid)

        best_param = copy.deepcopy(self.ADD_model.state_dict())
        save_path = get_or_create_path(save_path)
        torch.save(best_param, save_path)
        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.ADD_model.eval()
        x_values = x_test.values
        preds = []

        daily_index, daily_count = self.get_daily_inter(x_test, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            x_batch = torch.from_numpy(x_values[batch]).float().to(self.device)

            with torch.no_grad():
                pred = self.ADD_model(x_batch)
                pred = pred["excess"].detach().cpu().numpy()

            preds.append(pred)

        r = pd.Series(np.concatenate(preds), index=index)
        return r


class ADDModel(nn.Module):
    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
        dec_dropout=0.5,
        base_model="GRU",
        gamma=0.1,
        gamma_clip=0.4,
    ):
        super().__init__()
        self.d_feat = d_feat
        self.base_model = base_model
        if base_model == "GRU":
            self.enc_excess, self.enc_market = [
                nn.GRU(
                    input_size=d_feat,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout,
                )
                for _ in range(2)
            ]
        elif base_model == "LSTM":
            self.enc_excess, self.enc_market = [
                nn.LSTM(
                    input_size=d_feat,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout,
                )
                for _ in range(2)
            ]
        else:
            raise ValueError("unknown base model name `%s`" % base_model)
        self.dec = Decoder(d_feat, 2 * hidden_size, num_layers, dec_dropout, base_model)

        ctx_size = hidden_size * num_layers
        self.pred_excess, self.adv_excess = [
            nn.Sequential(nn.Linear(ctx_size, ctx_size), nn.BatchNorm1d(ctx_size), nn.Tanh(), nn.Linear(ctx_size, 1))
            for _ in range(2)
        ]
        self.adv_market, self.pred_market = [
            nn.Sequential(nn.Linear(ctx_size, ctx_size), nn.BatchNorm1d(ctx_size), nn.Tanh(), nn.Linear(ctx_size, 3))
            for _ in range(2)
        ]
        self.before_adv_market, self.before_adv_excess = [RevGrad(gamma, gamma_clip) for _ in range(2)]

    def forward(self, x):
        x = x.reshape(len(x), self.d_feat, -1)
        N = x.shape[0]
        T = x.shape[-1]
        x = x.permute(0, 2, 1)

        out, hidden_excess = self.enc_excess(x)
        out, hidden_market = self.enc_market(x)
        if self.base_model == "LSTM":
            feature_excess = hidden_excess[0].permute(1, 0, 2).reshape(N, -1)
            feature_market = hidden_market[0].permute(1, 0, 2).reshape(N, -1)
        else:
            feature_excess = hidden_excess.permute(1, 0, 2).reshape(N, -1)
            feature_market = hidden_market.permute(1, 0, 2).reshape(N, -1)
        predicts = {}
        predicts["excess"] = self.pred_excess(feature_excess).squeeze(1)
        predicts["market"] = self.pred_market(feature_market)
        predicts["adv_market"] = self.adv_market(self.before_adv_market(feature_excess))
        predicts["adv_excess"] = self.adv_excess(self.before_adv_excess(feature_market).squeeze(1))
        if self.base_model == "LSTM":
            hidden = [torch.cat([hidden_excess[i], hidden_market[i]], -1) for i in range(2)]
        else:
            hidden = torch.cat([hidden_excess, hidden_market], -1)
        x = torch.zeros_like(x[:, 1, :])
        reconstructed_feature = []
        for i in range(T):
            x, hidden = self.dec(x, hidden)
            reconstructed_feature.append(x)
        reconstructed_feature = torch.stack(reconstructed_feature, 1)
        predicts["reconstructed_feature"] = reconstructed_feature
        return predicts


class Decoder(nn.Module):
    def __init__(self, d_feat=6, hidden_size=128, num_layers=1, dropout=0.5, base_model="GRU"):
        super().__init__()
        self.base_model = base_model
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

        self.fc = nn.Linear(hidden_size, d_feat)

    def forward(self, x, hidden):
        x = x.unsqueeze(1)
        output, hidden = self.rnn(x, hidden)
        output = output.squeeze(1)
        pred = self.fc(output)
        return pred, hidden


class RevGradFunc(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


class RevGrad(nn.Module):
    def __init__(self, gamma=0.1, gamma_clip=0.4, *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self.gamma = gamma
        self.gamma_clip = torch.tensor(float(gamma_clip), requires_grad=False)
        self._alpha = torch.tensor(0, requires_grad=False)
        self._p = 0

    def step_alpha(self):
        self._p += 1
        self._alpha = min(
            self.gamma_clip, torch.tensor(2 / (1 + math.exp(-self.gamma * self._p)) - 1, requires_grad=False)
        )

    def forward(self, input_):
        return RevGradFunc.apply(input_, self._alpha)

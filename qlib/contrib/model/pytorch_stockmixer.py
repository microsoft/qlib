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


class StockMixer(Model):
    """StockMixer Model

    StockMixer is a simple yet strong MLP-based architecture for stock price
    forecasting (AAAI 2024). It captures complex correlations in stock data
    through three mixers -- indicator mixing, multi-scale time mixing and
    market-aware stock mixing -- without relying on any prior knowledge such
    as pre-defined stock graphs.

    Paper: StockMixer: A Simple yet Strong MLP-based Architecture for Stock
    Price Forecasting. https://ojs.aaai.org/index.php/AAAI/article/view/28681
    Official code: https://github.com/SJTU-DMTai/StockMixer

    .. note::

        The market-aware stock-mixing block of StockMixer requires the full
        cross-section of stocks of one trading day in a single batch. This
        wrapper therefore batches the tabular data produced by ``DatasetH``
        **day by day**: samples are grouped by the ``datetime`` level of the
        dataframe index, and each day is fed to the network as one
        ``(n_stock, time_steps, d_feat)`` tensor. Days with less than
        ``n_stock`` instruments are zero-padded (padded rows are masked out
        of the loss, excluded from the statistics of the market-aware
        stock-mixing layer, and dropped from the predictions); a day with
        more than ``n_stock`` instruments raises a ``ValueError``, so
        ``n_stock`` should be set to at least the maximum daily
        cross-section size (e.g. 305 for CSI300).

        For ``Alpha360`` data (360 = 6 indicators x 60 days) use
        ``d_feat=6, time_steps=60``, the setting closest to the paper. For
        ``Alpha158``-style snapshots without an explicit time axis use
        ``time_steps=1, d_feat=<num features>``: the multi-scale time-mixing
        branch then degenerates to a single-step linear mapping (a documented
        deviation from the paper), while indicator mixing and market-aware
        stock mixing are fully preserved.
    """

    def __init__(
        self,
        d_feat=6,
        time_steps=60,
        n_stock=305,
        market_dim=20,
        alpha=0.1,
        n_epochs=200,
        lr=0.001,
        metric="",
        early_stop=20,
        loss="mse",
        optimizer="adam",
        GPU=0,
        seed=None,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("StockMixer")
        self.logger.info("StockMixer pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.time_steps = time_steps
        self.n_stock = n_stock
        self.market_dim = market_dim
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed

        self.logger.info(
            "StockMixer parameters setting:"
            "\nd_feat : {}"
            "\ntime_steps : {}"
            "\nn_stock : {}"
            "\nmarket_dim : {}"
            "\nalpha : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\nvisible_GPU : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                time_steps,
                n_stock,
                market_dim,
                alpha,
                n_epochs,
                lr,
                metric,
                early_stop,
                optimizer.lower(),
                loss,
                GPU,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.stock_mixer_model = StockMixerModel(
            stocks=self.n_stock,
            channels=self.d_feat,
            time_steps=self.time_steps,
            market=self.market_dim,
        )
        self.logger.info("model:\n{:}".format(self.stock_mixer_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.stock_mixer_model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.stock_mixer_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.stock_mixer_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.stock_mixer_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)

    def rank_reg(self, pred, label):
        # pairwise ranking regularization as in the paper
        pre_pw_dif = pred.unsqueeze(-1) - pred.unsqueeze(0)
        gt_pw_dif = label.unsqueeze(0) - label.unsqueeze(-1)
        return torch.mean(torch.relu(pre_pw_dif * gt_pw_dif))

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            loss = self.mse(pred[mask], label[mask])
            if self.alpha > 0:
                loss = loss + self.alpha * self.rank_reg(pred[mask], label[mask])
            return loss

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def _prepare_day_groups(self, index):
        # group sample positions by the datetime level of the index
        dates = index.get_level_values("datetime")
        codes, _ = pd.factorize(dates, sort=True)
        order = np.argsort(codes, kind="stable")
        counts = np.bincount(codes)
        return np.split(order, np.cumsum(counts)[:-1])

    def _pad_day(self, x_day, y_day=None):
        # pad one day's cross-section to `n_stock` stocks
        n_day = x_day.shape[0]
        if n_day > self.n_stock:
            raise ValueError(
                "one day contains %d instruments, more than n_stock=%d; please increase "
                "`n_stock` to at least the maximum daily cross-section size" % (n_day, self.n_stock)
            )
        if n_day < self.n_stock:
            pad_x = np.zeros((self.n_stock - n_day, x_day.shape[1]), dtype=np.float32)
            x_day = np.concatenate([x_day, pad_x], axis=0)
            if y_day is not None:
                pad_y = np.full(self.n_stock - n_day, np.nan, dtype=np.float32)
                y_day = np.concatenate([y_day, pad_y], axis=0)
        return x_day, y_day, n_day

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)

        if x_train_values.shape[1] != self.d_feat * self.time_steps:
            raise ValueError(
                "the feature dimension of the data (%d) does not match d_feat * time_steps (%d)"
                % (x_train_values.shape[1], self.d_feat * self.time_steps)
            )

        self.stock_mixer_model.train()

        day_indices = self._prepare_day_groups(x_train.index)
        np.random.shuffle(day_indices)

        for day in day_indices:
            x_day, y_day, n_day = self._pad_day(x_train_values[day], y_train_values[day])

            feature = torch.from_numpy(x_day).float().to(self.device)
            label = torch.from_numpy(y_day).float().to(self.device)
            mask = torch.arange(self.n_stock, device=self.device) < n_day

            pred = self.stock_mixer_model(feature, mask)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.stock_mixer_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.stock_mixer_model.eval()

        scores = []
        losses = []

        day_indices = self._prepare_day_groups(data_x.index)

        for day in day_indices:
            x_day, y_day, n_day = self._pad_day(x_values[day], y_values[day])

            feature = torch.from_numpy(x_day).float().to(self.device)
            label = torch.from_numpy(y_day).float().to(self.device)
            mask = torch.arange(self.n_stock, device=self.device) < n_day

            with torch.no_grad():
                pred = self.stock_mixer_model(feature, mask)
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

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

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
                best_param = copy.deepcopy(self.stock_mixer_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.stock_mixer_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.stock_mixer_model.eval()
        x_values = x_test.values

        if x_values.shape[1] != self.d_feat * self.time_steps:
            raise ValueError(
                "the feature dimension of the data (%d) does not match d_feat * time_steps (%d)"
                % (x_values.shape[1], self.d_feat * self.time_steps)
            )

        preds = []

        day_indices = self._prepare_day_groups(index)

        for day in day_indices:
            x_day, _, n_day = self._pad_day(x_values[day])

            feature = torch.from_numpy(x_day).float().to(self.device)
            mask = torch.arange(self.n_stock, device=self.device) < n_day

            with torch.no_grad():
                pred = self.stock_mixer_model(feature, mask)[:n_day].detach().cpu().numpy()

            preds.append(pd.Series(pred, index=index[day]))

        return pd.concat(preds).sort_index()


class TriU(nn.Module):
    """Upper-triangular time mixing: every step only attends to past steps"""

    def __init__(self, time_steps):
        super().__init__()
        self.time_steps = time_steps
        self.triU = nn.ParameterList([nn.Linear(i + 1, 1) for i in range(time_steps)])

    def forward(self, inputs):
        # inputs: (stocks, channels, time_steps)
        x = self.triU[0](inputs[..., 0].unsqueeze(-1))
        for i in range(1, self.time_steps):
            x = torch.cat([x, self.triU[i](inputs[..., 0 : i + 1])], dim=-1)
        return x


class MixerBlock(nn.Module):
    """Simple two-layer MLP with GELU activation"""

    def __init__(self, mlp_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.dense_1 = nn.Linear(mlp_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dense_2 = nn.Linear(hidden_dim, mlp_dim)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        if self.dropout != 0.0:
            x = nn.functional.dropout(x, p=self.dropout)
        x = self.dense_2(x)
        if self.dropout != 0.0:
            x = nn.functional.dropout(x, p=self.dropout)
        return x


class Mixer2dTriU(nn.Module):
    """Alternate (upper-triangular) time mixing and indicator (channel) mixing"""

    def __init__(self, time_steps, channels):
        super().__init__()
        self.LN_1 = nn.LayerNorm([time_steps, channels])
        self.LN_2 = nn.LayerNorm([time_steps, channels])
        self.timeMixer = TriU(time_steps)
        self.channelMixer = MixerBlock(channels, channels)

    def forward(self, inputs):
        # inputs: (stocks, time_steps, channels)
        x = self.LN_1(inputs)
        x = x.permute(0, 2, 1)
        x = self.timeMixer(x)
        x = x.permute(0, 2, 1)

        x = self.LN_2(x + inputs)
        y = self.channelMixer(x)
        return x + y


class MultTime2dMixer(nn.Module):
    """Multi-scale time mixing: mix the raw series and a down-sampled series"""

    def __init__(self, time_steps, channels, scale_dim):
        super().__init__()
        self.mix_layer = Mixer2dTriU(time_steps, channels)
        self.scale_mix_layer = Mixer2dTriU(scale_dim, channels)

    def forward(self, inputs, y):
        # inputs: (stocks, time_steps, channels)
        # y: (stocks, scale_dim, channels), the down-sampled view of `inputs`
        y = self.scale_mix_layer(y)
        x = self.mix_layer(inputs)
        return torch.cat([inputs, x, y], dim=1)


class NoGraphMixer(nn.Module):
    """Market-aware stock mixing without any prior graph knowledge"""

    def __init__(self, stocks, hidden_dim=20):
        super().__init__()
        self.layer_norm_stock = nn.LayerNorm(stocks)
        self.dense1 = nn.Linear(stocks, hidden_dim)
        self.activation = nn.Hardswish()
        self.dense2 = nn.Linear(hidden_dim, stocks)

    def forward(self, inputs, mask=None):
        # inputs: (stocks, features)
        # mask: (stocks,) boolean tensor, True for real stocks, False for padding
        x = inputs
        x = x.permute(1, 0)
        if mask is None:
            x = self.layer_norm_stock(x)
        else:
            # masked layer-norm: statistics are computed over real stocks only,
            # so zero-padded rows cannot pollute the normalization
            x_real = x[:, mask]
            mean = x_real.mean(dim=-1, keepdim=True)
            var = x_real.var(dim=-1, keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(var + self.layer_norm_stock.eps)
            x = x * self.layer_norm_stock.weight + self.layer_norm_stock.bias
            # zero the padded rows so they do not leak through dense1
            x = x * mask.to(x.dtype)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        if mask is not None:
            # zero the padded rows again: dense2 mixes them back in otherwise
            x = x * mask.to(x.dtype)
        x = x.permute(1, 0)
        return x


class StockMixerModel(nn.Module):
    def __init__(self, stocks, channels, time_steps, market):
        super().__init__()
        self.channels = channels
        self.time_steps = time_steps
        # the multi-scale branch requires a window of length 2 at least; for
        # single-step data (e.g. Alpha158 with time_steps=1) it is disabled
        self.use_scale = time_steps >= 2
        self.scale_dim = time_steps // 2
        self.time_feat = time_steps * 2 + self.scale_dim if self.use_scale else time_steps

        if self.use_scale:
            self.mixer = MultTime2dMixer(time_steps, channels, scale_dim=self.scale_dim)
            self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=2)
        else:
            self.mix_layer = Mixer2dTriU(time_steps, channels)
        self.channel_fc = nn.Linear(channels, 1)
        self.stock_mixer = NoGraphMixer(stocks, market)
        self.time_fc = nn.Linear(self.time_feat, 1)
        self.time_fc_ = nn.Linear(self.time_feat, 1)

    def forward(self, inputs, mask=None):
        # inputs: (stocks, channels * time_steps) tabular rows (Alpha360-style layout)
        # mask: (stocks,) boolean tensor, True for real stocks, False for padding
        x = inputs.reshape(inputs.shape[0], self.channels, self.time_steps).permute(0, 2, 1)

        if self.use_scale:
            scale = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
            y = self.mixer(x, scale)
        else:
            y = self.mix_layer(x)
        y = self.channel_fc(y).squeeze(-1)

        z = self.stock_mixer(y, mask)
        y = self.time_fc(y)
        z = self.time_fc_(z)
        return (y + z).squeeze(-1)

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
from ...utils import (
    unpack_archive_with_buffer,
    save_multiple_parts_file,
    create_save_path,
    drop_nan_by_y_index,
)
from ...log import get_module_logger, TimeInspector

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP


class TabnetModel(Model):
    def __init__(
        self,
        d_feat=158,
        out_dim=64,
        final_out_dim=1,
        batch_size=4096,
        n_d=64,
        n_a=64,
        n_shared=2,
        n_ind=2,
        n_steps=5,
        n_epochs=100,
        pretrain_n_epochs=50,
        relax=1.3,
        vbs=2048,
        seed=993,
        optimizer="adam",
        loss="mse",
        metric="",
        early_stop=20,
        GPU="1",
        pretrain_loss="custom",
        ps=0.3,
        lr=0.01,
        pretrain=True,
        pretrain_file="./pretrain/best.model",
    ):
        """
        TabNet model for Qlib

        Args：
        ps: probability to generate the bernoulli mask
        """
        # set hyper-parameters.
        self.d_feat = d_feat
        self.out_dim = out_dim
        self.final_out_dim = final_out_dim
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = optimizer.lower()
        self.pretrain_loss = pretrain_loss
        self.seed = seed
        self.ps = ps
        self.n_epochs = n_epochs
        self.logger = get_module_logger("TabNet")
        self.pretrain_n_epochs = pretrain_n_epochs
        self.device = "cuda:%s" % (GPU) if torch.cuda.is_available() else "cpu"
        self.loss = loss
        self.metric = metric
        self.early_stop = early_stop
        self.pretrain = pretrain
        self.pretrain_file = pretrain_file
        self.logger.info(
            "TabNet:"
            "\nbatch_size : {}"
            "\nvirtual bs : {}"
            "\nGPU : {}"
            "\npretrain: {}".format(self.batch_size, vbs, GPU, pretrain)
        )
        self.fitted = False
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.tabnet_model = TabNet(
            inp_dim=self.d_feat, out_dim=self.out_dim, vbs=vbs, relax=relax, device=self.device
        ).to(self.device)
        self.tabnet_decoder = TabNet_Decoder(self.out_dim, self.d_feat, n_shared, n_ind, vbs, n_steps, self.device).to(
            self.device
        )

        if optimizer.lower() == "adam":
            self.pretrain_optimizer = optim.Adam(
                list(self.tabnet_model.parameters()) + list(self.tabnet_decoder.parameters()), lr=self.lr
            )
            self.train_optimizer = optim.Adam(self.tabnet_model.parameters(), lr=self.lr)

        elif optimizer.lower() == "gd":
            self.pretrain_optimizer = optim.SGD(
                list(self.tabnet_model.parameters()) + list(self.tabnet_decoder.parameters()), lr=self.lr
            )
            self.train_optimizer = optim.SGD(self.tabnet_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

    def pretrain_fn(self, dataset=DatasetH, pretrain_file="./pretrain/best.model"):
        # make a directory if pretrian director does not exist
        if pretrain_file.startswith("./pretrain") and not os.path.exists("pretrain"):
            self.logger.info("make folder to store model...")
            os.makedirs("pretrain")

        [df_train, df_valid] = dataset.prepare(
            ["pretrain", "pretrain_validation"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )

        df_train.fillna(df_train.mean(), inplace=True)
        df_valid.fillna(df_valid.mean(), inplace=True)

        x_train = df_train["feature"]
        x_valid = df_valid["feature"]

        # Early stop setup
        stop_steps = 0
        train_loss = 0
        best_loss = np.inf

        for epoch_idx in range(self.pretrain_n_epochs):
            self.logger.info("epoch: %s" % (epoch_idx))
            self.logger.info("pre-training...")
            self.pretrain_epoch(x_train)
            self.logger.info("evaluating...")
            train_loss = self.pretrain_test_epoch(x_train)
            valid_loss = self.pretrain_test_epoch(x_valid)
            self.logger.info("train %.6f, valid %.6f" % (train_loss, valid_loss))

            if valid_loss < best_loss:
                self.logger.info("Save Model...")
                torch.save(self.tabnet_model.state_dict(), pretrain_file)
                best_loss = valid_loss
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        verbose=True,
        save_path=None,
    ):
        if self.pretrain:
            # there is a  pretrained model, load the model
            self.logger.info("Pretrain...")
            self.pretrain_fn(dataset, self.pretrain_file)
            self.logger.info("Load Pretrain model")
            self.tabnet_model.load_state_dict(torch.load(self.pretrain_file))

        # adding one more linear layer to fit the final output dimension
        self.tabnet_model = FinetuneModel(self.out_dim, self.final_out_dim, self.tabnet_model).to(self.device)
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        df_train.fillna(df_train.mean(), inplace=True)
        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        stop_steps = 0
        train_loss = 0
        best_score = np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        self.logger.info("training...")
        self.fitted = True

        for epoch_idx in range(self.n_epochs):
            self.logger.info("epoch: %s" % (epoch_idx))
            self.logger.info("training...")
            self.train_epoch(x_train, y_train)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(x_train, y_train)
            valid_loss, val_score = self.test_epoch(x_valid, y_valid)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score < best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = epoch_idx
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break
        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.tabnet_model.eval()
        x_values = torch.from_numpy(x_test.values)
        x_values[torch.isnan(x_values)] = 0
        sample_num = x_values.shape[0]
        preds = []

        for begin in range(sample_num)[:: self.batch_size]:
            if sample_num - begin < self.batch_size:
                end = sample_num
            else:
                end = begin + self.batch_size

            x_batch = x_values[begin:end].float().to(self.device)
            priors = torch.ones(end - begin, self.d_feat).to(self.device)

            with torch.no_grad():
                pred = self.tabnet_model(x_batch, priors).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)

    def test_epoch(self, data_x, data_y):
        # prepare training data
        x_values = torch.from_numpy(data_x.values)
        y_values = torch.from_numpy(np.squeeze(data_y.values))
        x_values[torch.isnan(x_values)] = 0
        y_values[torch.isnan(y_values)] = 0
        self.tabnet_model.eval()

        scores = []
        losses = []

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:

            if len(indices) - i < self.batch_size:
                break
            feature = x_values[indices[i : i + self.batch_size]].float().to(self.device)
            label = y_values[indices[i : i + self.batch_size]].float().to(self.device)
            priors = torch.ones(self.batch_size, self.d_feat).to(self.device)
            pred = self.tabnet_model(feature, priors)
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            score = self.metric_fn(pred, label)
            scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def train_epoch(self, x_train, y_train):
        x_train_values = torch.from_numpy(x_train.values)
        y_train_values = torch.from_numpy(np.squeeze(y_train.values))
        x_train_values[torch.isnan(x_train_values)] = 0
        y_train_values[torch.isnan(y_train_values)] = 0
        self.tabnet_model.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:

            if len(indices) - i < self.batch_size:
                break

            feature = x_train_values[indices[i : i + self.batch_size]].float().to(self.device)
            label = y_train_values[indices[i : i + self.batch_size]].float().to(self.device)
            priors = torch.ones(self.batch_size, self.d_feat).to(self.device)
            pred = self.tabnet_model(feature, priors)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.tabnet_model.parameters(), 3.0)
            self.train_optimizer.step()

    def pretrain_epoch(self, x_train):
        train_set = torch.from_numpy(x_train.values)
        train_set[torch.isnan(train_set)] = 0
        indices = np.arange(len(train_set))
        np.random.shuffle(indices)

        self.tabnet_model.train()
        self.tabnet_decoder.train()

        for i in range(len(indices))[:: self.batch_size]:

            if len(indices) - i < self.batch_size:
                break

            S_mask = torch.bernoulli(torch.empty(self.batch_size, self.d_feat).fill_(self.ps))
            x_train_values = train_set[indices[i : i + self.batch_size]] * (1 - S_mask)
            y_train_values = train_set[indices[i : i + self.batch_size]] * (S_mask)

            S_mask = S_mask.to(self.device)
            feature = x_train_values.float().to(self.device)
            label = y_train_values.float().to(self.device)
            priors = 1 - S_mask
            (vec, sparse_loss) = self.tabnet_model(feature, priors)
            f = self.tabnet_decoder(vec)
            loss = self.pretrain_loss_fn(label, f, S_mask)

            self.pretrain_optimizer.zero_grad()
            loss.backward()
            self.pretrain_optimizer.step()

    def pretrain_test_epoch(self, x_train):
        train_set = torch.from_numpy(x_train.values)
        train_set[torch.isnan(train_set)] = 0
        indices = np.arange(len(train_set))

        self.tabnet_model.eval()
        self.tabnet_decoder.eval()

        losses = []

        for i in range(len(indices))[:: self.batch_size]:

            if len(indices) - i < self.batch_size:
                break

            S_mask = torch.bernoulli(torch.empty(self.batch_size, self.d_feat).fill_(self.ps))
            x_train_values = train_set[indices[i : i + self.batch_size]] * (1 - S_mask)
            y_train_values = train_set[indices[i : i + self.batch_size]] * (S_mask)

            feature = x_train_values.float().to(self.device)
            label = y_train_values.float().to(self.device)
            S_mask = S_mask.to(self.device)
            priors = 1 - S_mask
            (vec, sparse_loss) = self.tabnet_model(feature, priors)
            f = self.tabnet_decoder(vec)

            loss = self.pretrain_loss_fn(label, f, S_mask)
            losses.append(loss.item())

        return np.mean(losses)

    def pretrain_loss_fn(self, f_hat, f, S):
        """
        Pretrain loss function defined in the original paper, read "Tabular self-supervised learning" in https://arxiv.org/pdf/1908.07442.pdf
        """
        down_mean = torch.mean(f, dim=0)
        down = torch.sqrt(torch.sum(torch.square(f - down_mean), dim=0))
        up = (f_hat - f) * S
        return torch.sum(torch.square(up / down))

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

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)


class FinetuneModel(nn.Module):
    """
    FinuetuneModel for adding a layer by the end
    """

    def __init__(self, input_dim, output_dim, trained_model):
        super().__init__()
        self.model = trained_model
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x, priors):
        return self.fc(self.model(x, priors)[0]).squeeze()  # take the vec out


class DecoderStep(nn.Module):
    def __init__(self, inp_dim, out_dim, shared, n_ind, vbs, device):
        super().__init__()
        self.fea_tran = FeatureTransformer(inp_dim, out_dim, shared, n_ind, vbs, device)
        self.fc = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        x = self.fea_tran(x)
        return self.fc(x)


class TabNet_Decoder(nn.Module):
    def __init__(self, inp_dim, out_dim, n_shared, n_ind, vbs, n_steps, device):
        """
        TabNet decoder that is used in pre-training
        """
        self.out_dim = out_dim

        super().__init__()
        if n_shared > 0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim, 2 * out_dim))
            for x in range(n_shared - 1):
                self.shared.append(nn.Linear(out_dim, 2 * out_dim))  # preset the linear function we will use
        else:
            self.shared = None
        self.n_steps = n_steps
        self.steps = nn.ModuleList()
        for x in range(n_steps):
            self.steps.append(DecoderStep(inp_dim, out_dim, self.shared, n_ind, vbs, device))

    def forward(self, x):
        out = torch.zeros(x.size(0), self.out_dim).to(x.device)
        for step in self.steps:
            out += step(x)
        return out


class TabNet(nn.Module):
    def __init__(
        self, inp_dim=6, out_dim=6, n_d=64, n_a=64, n_shared=2, n_ind=2, n_steps=5, relax=1.2, vbs=1024, device="cpu"
    ):
        """
        TabNet AKA the original encoder

        Args:
            n_d: dimension of the features used to calculate the final results
            n_a: dimension of the features input to the attention transformer of the next step
            n_shared: numbr of shared steps in feature transfomer(optional)
            n_ind: number of independent steps in feature transformer
            n_steps: number of steps of pass through tabbet
            relax coefficient:
            virtual batch size:
        """
        super().__init__()

        # set the number of shared step in feature transformer
        if n_shared > 0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim, 2 * (n_d + n_a)))
            for x in range(n_shared - 1):
                self.shared.append(nn.Linear(n_d + n_a, 2 * (n_d + n_a)))  # preset the linear function we will use
        else:
            self.shared = None

        self.first_step = FeatureTransformer(inp_dim, n_d + n_a, self.shared, n_ind, vbs, device)
        self.steps = nn.ModuleList()
        for x in range(n_steps - 1):
            self.steps.append(DecisionStep(inp_dim, n_d, n_a, self.shared, n_ind, relax, vbs, device))
        self.fc = nn.Linear(n_d, out_dim)
        self.bn = nn.BatchNorm1d(inp_dim, momentum=0.01)
        self.n_d = n_d

    def forward(self, x, priors):
        assert not torch.isnan(x).any()
        x = self.bn(x)
        x_a = self.first_step(x)[:, self.n_d :]
        sparse_loss = torch.zeros(1).to(x.device)
        out = torch.zeros(x.size(0), self.n_d).to(x.device)
        for step in self.steps:
            x_te, l = step(x, x_a, priors)
            out += F.relu(x_te[:, : self.n_d])  # split the feautre from feat_transformer
            x_a = x_te[:, self.n_d :]
            sparse_loss += l
        return self.fc(out), sparse_loss


class GBN(nn.Module):
    """
    Ghost Batch Normalization
    an efficient way of doing batch normalization

    Args:
        vbs: virtual batch size
    """

    def __init__(self, inp, vbs=1024, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm1d(inp, momentum=momentum)
        self.vbs = vbs

    def forward(self, x):
        chunk = torch.chunk(x, x.size(0) // self.vbs, 0)
        res = [self.bn(y) for y in chunk]
        return torch.cat(res, 0)


class GLU(nn.Module):
    """
    GLU block that extracts only the most essential information

    Args:
        vbs: virtual batch size
    """

    def __init__(self, inp_dim, out_dim, fc=None, vbs=1024):
        super().__init__()
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(inp_dim, out_dim * 2)
        self.bn = GBN(out_dim * 2, vbs=vbs)
        self.od = out_dim

    def forward(self, x):
        x = self.bn(self.fc(x))
        return torch.mul(x[:, : self.od], torch.sigmoid(x[:, self.od :]))


class AttentionTransformer(nn.Module):
    """
    Args:
        relax: relax coefficient. The greater it is, we can
        use the same features more. When it is set to 1
        we can use every feature only once
    """

    def __init__(self, d_a, inp_dim, relax, vbs=1024):
        super().__init__()
        self.fc = nn.Linear(d_a, inp_dim)
        self.bn = GBN(inp_dim, vbs=vbs)
        self.r = relax

    # a:feature from previous decision step
    def forward(self, a, priors):
        a = self.bn(self.fc(a))
        mask = SparsemaxFunction.apply(a * priors)
        priors = priors * (self.r - mask)  # updating the prior
        return mask


class FeatureTransformer(nn.Module):
    def __init__(self, inp_dim, out_dim, shared, n_ind, vbs, device):
        super().__init__()
        first = True
        self.shared = nn.ModuleList()
        if shared:
            self.shared.append(GLU(inp_dim, out_dim, shared[0], vbs=vbs))
            first = False
            for fc in shared[1:]:
                self.shared.append(GLU(out_dim, out_dim, fc, vbs=vbs))
        else:
            self.shared = None
        self.independ = nn.ModuleList()
        if first:
            self.independ.append(GLU(inp, out_dim, vbs=vbs))
        for x in range(first, n_ind):
            self.independ.append(GLU(out_dim, out_dim, vbs=vbs))
        self.scale = torch.sqrt(torch.tensor([0.5], device=device))

    def forward(self, x):
        if self.shared:
            x = self.shared[0](x)
            for glu in self.shared[1:]:
                x = torch.add(x, glu(x))
                x = x * self.scale
        for glu in self.independ:
            x = torch.add(x, glu(x))
            x = x * self.scale
        return x


class DecisionStep(nn.Module):
    """
    One step for the TabNet
    """

    def __init__(self, inp_dim, n_d, n_a, shared, n_ind, relax, vbs, device):
        super().__init__()
        self.atten_tran = AttentionTransformer(n_a, inp_dim, relax, vbs)
        self.fea_tran = FeatureTransformer(inp_dim, n_d + n_a, shared, n_ind, vbs, device)

    def forward(self, x, a, priors):
        mask = self.atten_tran(a, priors)
        sparse_loss = ((-1) * mask * torch.log(mask + 1e-10)).mean()
        x = self.fea_tran(x * mask)
        return x, sparse_loss


def make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


class SparsemaxFunction(Function):
    """
    SparseMax function for replacing reLU
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = SparsemaxFunction.threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

    @staticmethod
    def threshold_and_support(input, dim=-1):
        input_srt, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum

        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input.dtype)
        return tau, support_size

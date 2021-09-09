# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import copy
import math
import json
import collections
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    SummaryWriter = None

from tqdm import tqdm

from qlib.utils import get_or_create_path
from qlib.log import get_module_logger
from qlib.model.base import Model
from qlib.contrib.data.dataset import MTSDatasetH

device = "cuda" if torch.cuda.is_available() else "cpu"


class TRAModel(Model):
    """
    TRA Model

    Args:
        model_config (dict): model config (will be used by RNN or Transformer)
        tra_config (dict): TRA config (will be used by TRA)
        model_type (str): which backbone model to use (RNN/Transformer)
        lr (float): learning rate
        n_epochs (int): number of total epochs
        early_stop (int): early stop when performance not improved at this step
        smooth_steps (int): number of steps for parameter smoothing
        max_steps_per_epoch (int): maximum number of steps in one epoch
        lamb (float): regularization parameter
        rho (float): exponential decay rate for `lamb`
        seed (int): random seed
        logdir (str): local log directory
        eval_train (bool): whether evaluate train set between epochs
        eval_test (bool): whether evaluate test set between epochs
        pretrain (bool): whether pretrain the backbone model before training TRA.
            Note that only TRA will be optimized after pretraining
        init_state (str): model init state path
        freeze_model (bool): whether freeze backbone model parameters
        freeze_predictors (bool): whether freeze predictors parameters
        transport_method (str): transport method, can be none/router/oracle
        memory_mode (str): memory mode, the same argument for MTSDatasetH
    """

    def __init__(
        self,
        model_config,
        tra_config,
        model_type="RNN",
        lr=1e-3,
        n_epochs=500,
        early_stop=50,
        smooth_steps=5,
        max_steps_per_epoch=None,
        lamb=0.0,
        rho=0.99,
        seed=0,
        logdir=None,
        eval_train=False,
        eval_test=False,
        pretrain=False,
        init_state=None,
        freeze_model=False,
        freeze_predictors=False,
        transport_method="none",
        memory_mode="sample",
    ):

        self.logger = get_module_logger("TRA")

        assert memory_mode in ["sample", "daily"], "invalid memory mode"
        assert transport_method in ["none", "router", "oracle"], f"invalid transport method {transport_method}"
        assert transport_method == "none" or tra_config["num_states"] > 1, "optimal transport requires `num_states` > 1"
        assert (
            memory_mode != "daily" or tra_config["src_info"] == "TPE"
        ), "daily transport can only support TPE as `src_info`"

        if transport_method == "router" and not eval_train:
            self.logger.warning("`eval_train` will be ignored when using TRA.router")

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.model_config = model_config
        self.tra_config = tra_config
        self.model_type = model_type
        self.lr = lr
        self.n_epochs = n_epochs
        self.early_stop = early_stop
        self.smooth_steps = smooth_steps
        self.max_steps_per_epoch = max_steps_per_epoch
        self.lamb = lamb
        self.rho = rho
        self.seed = seed
        self.logdir = logdir
        self.eval_train = eval_train
        self.eval_test = eval_test
        self.pretrain = pretrain
        self.init_state = init_state
        self.freeze_model = freeze_model
        self.freeze_predictors = freeze_predictors
        self.transport_method = transport_method
        self.use_daily_transport = memory_mode == "daily"
        self.transport_fn = transport_daily if self.use_daily_transport else transport_sample

        self._writer = None
        if self.logdir is not None:
            if os.path.exists(self.logdir):
                self.logger.warning(f"logdir {self.logdir} is not empty")
            os.makedirs(self.logdir, exist_ok=True)
            if SummaryWriter is not None:
                self._writer = SummaryWriter(log_dir=self.logdir)

        self._init_model()

    def _init_model(self):

        self.logger.info("init TRAModel...")

        self.model = eval(self.model_type)(**self.model_config).to(device)
        print(self.model)

        self.tra = TRA(self.model.output_size, **self.tra_config).to(device)
        print(self.tra)

        if self.init_state:
            self.logger.warninging(f"load state dict from `init_state`")
            state_dict = torch.load(self.init_state, map_location="cpu")
            self.model.load_state_dict(state_dict["model"])
            try:
                self.tra.load_state_dict(state_dict["tra"])
            except:
                self.logger.warninging("cannot load tra model, will skip")

        if self.freeze_model:
            self.logger.warninging(f"freeze model parameters")
            for param in self.model.parameters():
                param.requires_grad_(False)
        if self.freeze_predictors:
            self.logger.warninging(f"freeze TRA.predictors parameters")
            for param in self.tra.predictors.parameters():
                param.requires_grad_(False)

        self.logger.info("# model params: %d" % sum([p.numel() for p in self.model.parameters() if p.requires_grad]))
        self.logger.info("# tra params: %d" % sum([p.numel() for p in self.tra.parameters() if p.requires_grad]))

        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.tra.parameters()), lr=self.lr)

        self.fitted = False
        self.global_step = -1

    def train_epoch(self, epoch, data_set, is_pretrain=False):

        self.model.train()
        self.tra.train()
        data_set.train()

        max_steps = len(data_set)
        if self.max_steps_per_epoch is not None:
            if epoch == 0 and self.max_steps_per_epoch < max_steps:
                self.logger.info(f"max steps updated from {max_steps} to {self.max_steps_per_epoch}")
            max_steps = min(self.max_steps_per_epoch, max_steps)

        cur_step = 0
        total_loss = 0
        total_count = 0
        for batch in tqdm(data_set, total=max_steps):
            cur_step += 1
            if cur_step > max_steps:
                break

            self.global_step += 1

            data, state, label, count = batch["data"], batch["state"], batch["label"], batch["daily_count"]
            index = batch["daily_index"] if self.use_daily_transport else batch["index"]

            hidden = self.model(data)
            all_preds, choice, prob = self.tra(hidden, state)

            if not is_pretrain and self.transport_method != "none":
                loss, pred, L, P = self.transport_fn(
                    all_preds, label, choice, prob, count, self.transport_method, training=True
                )
                data_set.assign_data(index, L)  # save loss to memory
                lamb = self.lamb * (self.rho ** self.global_step)  # regularization decay
                reg = prob.log().mul(P).sum(dim=1).mean()  # train router to predict OT assignment
                if self._writer is not None:
                    self._writer.add_scalar("training/router_loss", -reg.item(), self.global_step)
                    self._writer.add_scalar("training/reg_loss", loss.item(), self.global_step)
                    self._writer.add_scalar("training/lamb", lamb, self.global_step)
                    prob_mean = prob.mean(axis=0).detach()
                    self._writer.add_scalar("training/prob_max", prob_mean.max(), self.global_step)
                    self._writer.add_scalar("training/prob_min", prob_mean.min(), self.global_step)
                    P_mean = P.mean(axis=0).detach()
                    self._writer.add_scalar("training/P_max", P_mean.max(), self.global_step)
                    self._writer.add_scalar("training/P_min", P_mean.min(), self.global_step)
                loss = loss - lamb * reg
            else:
                pred = all_preds.mean(dim=1)
                loss = loss_fn(pred, label)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self._writer is not None:
                self._writer.add_scalar("training/total_loss", loss.item(), self.global_step)

            total_loss += loss.item()
            total_count += 1

        total_loss /= total_count

        if self._writer is not None:
            self._writer.add_scalar("training/loss", total_loss, epoch)

        return total_loss

    def test_epoch(self, epoch, data_set, return_pred=False, prefix="test", is_pretrain=False):

        self.model.eval()
        self.tra.eval()
        data_set.eval()

        preds = []
        probs = []
        metrics = []
        for batch in tqdm(data_set):
            data, state, label, count = batch["data"], batch["state"], batch["label"], batch["daily_count"]
            index = batch["daily_index"] if self.use_daily_transport else batch["index"]

            with torch.no_grad():
                hidden = self.model(data)
                all_preds, choice, prob = self.tra(hidden, state)

            if not is_pretrain and self.transport_method != "none":
                loss, pred, L, P = self.transport_fn(
                    all_preds, label, choice, prob, count, self.transport_method, training=False
                )
                data_set.assign_data(index, L)  # save loss to memory
            else:
                pred = all_preds.mean(dim=1)

            X = np.c_[pred.cpu().numpy(), label.cpu().numpy(), all_preds.cpu().numpy()]
            columns = ["score", "label"] + ["score_%d" % d for d in range(all_preds.shape[1])]
            pred = pd.DataFrame(X, index=batch["index"], columns=columns)

            metrics.append(evaluate(pred))

            if return_pred:
                preds.append(pred)
                if prob is not None:
                    columns = ["prob_%d" % d for d in range(all_preds.shape[1])]
                    probs.append(pd.DataFrame(prob.cpu().numpy(), index=index, columns=columns))

        metrics = pd.DataFrame(metrics)
        metrics = {
            "MSE": metrics.MSE.mean(),
            "MAE": metrics.MAE.mean(),
            "IC": metrics.IC.mean(),
            "ICIR": metrics.IC.mean() / metrics.IC.std(),
        }

        if self._writer is not None and epoch >= 0:
            for key, value in metrics.items():
                self._writer.add_scalar(prefix + "/" + key, value, epoch)

        if return_pred:
            preds = pd.concat(preds, axis=0)
            preds.index = data_set.restore_index(preds.index)
            preds.index = preds.index.swaplevel()
            preds.sort_index(inplace=True)
            if probs:
                probs = pd.concat(probs, axis=0)
                if self.use_daily_transport:
                    probs.index = data_set.restore_daily_index(probs.index)
                else:
                    probs.index = data_set.restore_index(probs.index)
                    probs.index = probs.index.swaplevel()
                    probs.sort_index(inplace=True)

        return metrics, preds, probs

    def _fit(self, train_set, valid_set, test_set, evals_result, start_epoch=0, is_pretrain=True):

        best_score = -1
        best_epoch = 0
        stop_rounds = 0
        best_params = {
            "model": copy.deepcopy(self.model.state_dict()),
            "tra": copy.deepcopy(self.tra.state_dict()),
        }
        params_list = {
            "model": collections.deque(maxlen=self.smooth_steps),
            "tra": collections.deque(maxlen=self.smooth_steps),
        }

        # train
        if not is_pretrain and self.transport_method == "router":
            self.logger.info("init memory...")
            self.test_epoch(-1, train_set)

        for epoch in range(start_epoch, start_epoch + self.n_epochs):
            self.logger.info("Epoch %d:", epoch)

            self.logger.info("training...")
            self.train_epoch(epoch, train_set, is_pretrain=is_pretrain)

            self.logger.info("evaluating...")
            # average params for inference
            params_list["model"].append(copy.deepcopy(self.model.state_dict()))
            params_list["tra"].append(copy.deepcopy(self.tra.state_dict()))
            self.model.load_state_dict(average_params(params_list["model"]))
            self.tra.load_state_dict(average_params(params_list["tra"]))

            # NOTE: during evaluating, the whole memory will be refreshed
            if not is_pretrain and (self.transport_method == "router" or self.eval_train):
                train_set.clear_memory()  # NOTE: clear the shared memory
                train_metrics = self.test_epoch(epoch, train_set, is_pretrain=is_pretrain, prefix="train")[0]
                evals_result["train"].append(train_metrics)
                self.logger.info("train metrics: %s" % train_metrics)

            valid_metrics = self.test_epoch(epoch, valid_set, is_pretrain=is_pretrain, prefix="valid")[0]
            evals_result["valid"].append(valid_metrics)
            self.logger.info("valid metrics: %s" % valid_metrics)

            if self.eval_test:
                test_metrics = self.test_epoch(epoch, test_set, is_pretrain=is_pretrain, prefix="test")[0]
                evals_result["test"].append(test_metrics)
                self.logger.info("test metrics: %s" % test_metrics)

            if valid_metrics["IC"] > best_score:
                best_score = valid_metrics["IC"]
                stop_rounds = 0
                best_epoch = epoch
                best_params = {
                    "model": copy.deepcopy(self.model.state_dict()),
                    "tra": copy.deepcopy(self.tra.state_dict()),
                }
                if self.logdir is not None:
                    torch.save(best_params, self.logdir + "/model.bin")
            else:
                stop_rounds += 1
                if stop_rounds >= self.early_stop:
                    self.logger.info("early stop @ %s" % epoch)
                    break

            # restore parameters
            self.model.load_state_dict(params_list["model"][-1])
            self.tra.load_state_dict(params_list["tra"][-1])

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_params["model"])
        self.tra.load_state_dict(best_params["tra"])

        return best_score, epoch

    def fit(self, dataset, evals_result=dict()):

        assert isinstance(dataset, MTSDatasetH), "TRAModel only supports `qlib.contrib.data.dataset.MTSDatasetH`"

        train_set, valid_set, test_set = dataset.prepare(["train", "valid", "test"])

        self.fitted = True
        self.global_step = -1

        evals_result["train"] = []
        evals_result["valid"] = []
        evals_result["test"] = []

        epoch = 0
        if self.pretrain:

            self.logger.info("pretraining...")
            self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.tra.parameters()), lr=self.lr)
            _, epoch = self._fit(train_set, valid_set, test_set, evals_result, is_pretrain=True)

            self.logger.info("reset TRA")
            self.tra.reset_parameters()  # reset both router and predictors

            self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.tra.parameters()), lr=self.lr)

        self.logger.info("training...")
        best_score, _ = self._fit(train_set, valid_set, test_set, evals_result, start_epoch=epoch, is_pretrain=False)

        self.logger.info("inference")
        train_metrics, train_preds, train_probs = self.test_epoch(-1, train_set, return_pred=True)
        self.logger.info("train metrics: %s" % train_metrics)

        valid_metrics, valid_preds, valid_probs = self.test_epoch(-1, valid_set, return_pred=True)
        self.logger.info("valid metrics: %s" % valid_metrics)

        test_metrics, test_preds, test_probs = self.test_epoch(-1, test_set, return_pred=True)
        self.logger.info("test metrics: %s" % test_metrics)

        if self.logdir:
            self.logger.info("save model & pred to local directory")

            pd.concat({name: pd.DataFrame(evals_result[name]) for name in evals_result}, axis=1).to_csv(
                self.logdir + "/logs.csv", index=False
            )

            torch.save({"model": self.model.state_dict(), "tra": self.tra.state_dict()}, self.logdir + "/model.bin")

            train_preds.to_pickle(self.logdir + "/train_pred.pkl")
            valid_preds.to_pickle(self.logdir + "/valid_pred.pkl")
            test_preds.to_pickle(self.logdir + "/test_pred.pkl")

            if len(train_probs):
                train_probs.to_pickle(self.logdir + "/train_prob.pkl")
                valid_probs.to_pickle(self.logdir + "/valid_prob.pkl")
                test_probs.to_pickle(self.logdir + "/test_prob.pkl")

            info = {
                "config": {
                    "model_config": self.model_config,
                    "tra_config": self.tra_config,
                    "model_type": self.model_type,
                    "lr": self.lr,
                    "n_epochs": self.n_epochs,
                    "early_stop": self.early_stop,
                    "smooth_steps": self.smooth_steps,
                    "max_steps_per_epoch": self.max_steps_per_epoch,
                    "lamb": self.lamb,
                    "rho": self.rho,
                    "seed": self.seed,
                    "logdir": self.logdir,
                    "pretrain": self.pretrain,
                    "init_state": self.init_state,
                    "transport_method": self.transport_method,
                    "use_daily_transport": self.use_daily_transport,
                },
                "best_eval_metric": -best_score,  # NOTE: -1 for minimize
                "metrics": {"train": train_metrics, "valid": valid_metrics, "test": test_metrics},
            }
            with open(self.logdir + "/info.json", "w") as f:
                json.dump(info, f)

    def predict(self, dataset, segment="test"):

        assert isinstance(dataset, MTSDatasetH), "TRAModel only supports `qlib.contrib.data.dataset.MTSDatasetH`"

        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        test_set = dataset.prepare(segment)

        metrics, preds, probs = self.test_epoch(-1, test_set, return_pred=True)
        self.logger.info("test metrics: %s" % metrics)

        return preds


class RNN(nn.Module):

    """RNN Model

    Args:
        input_size (int): input size (# features)
        hidden_size (int): hidden size
        num_layers (int): number of hidden layers
        rnn_arch (str): rnn architecture
        use_attn (bool): whether use attention layer.
            we use concat attention as https://github.com/fulifeng/Adv-AGRU/
        dropout (float): dropout rate
    """

    def __init__(
        self,
        input_size=16,
        hidden_size=64,
        num_layers=2,
        rnn_arch="GRU",
        use_attn=True,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_arch = rnn_arch
        self.use_attn = use_attn

        self.input_proj = nn.Linear(input_size, hidden_size)

        self.rnn = getattr(nn, rnn_arch)(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        if self.use_attn:
            self.W = nn.Linear(hidden_size, hidden_size)
            self.u = nn.Linear(hidden_size, 1, bias=False)
            self.output_size = hidden_size * 2
        else:
            self.output_size = hidden_size

    def forward(self, x):

        x = self.input_proj(x)

        rnn_out, last_out = self.rnn(x)
        if self.rnn_arch == "LSTM":
            last_out = last_out[0]
        last_out = last_out.mean(dim=0)

        if self.use_attn:
            laten = self.W(rnn_out).tanh()
            scores = self.u(laten).softmax(dim=1)
            att_out = (rnn_out * scores).sum(dim=1)
            last_out = torch.cat([last_out, att_out], dim=1)

        return last_out


class PositionalEncoding(nn.Module):
    # reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):

    """Transformer Model

    Args:
        input_size (int): input size (# features)
        hidden_size (int): hidden size
        num_layers (int): number of transformer layers
        num_heads (int): number of heads in transformer
        dropout (float): dropout rate
    """

    def __init__(
        self,
        input_size=16,
        hidden_size=64,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.input_proj = nn.Linear(input_size, hidden_size)

        self.pe = PositionalEncoding(input_size, dropout)
        layer = nn.TransformerEncoderLayer(
            nhead=num_heads, dropout=dropout, d_model=hidden_size, dim_feedforward=hidden_size * 4
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.output_size = hidden_size

    def forward(self, x):

        x = x.permute(1, 0, 2).contiguous()  # the first dim need to be time
        x = self.pe(x)

        x = self.input_proj(x)
        out = self.encoder(x)

        return out[-1]


class TRA(nn.Module):

    """Temporal Routing Adaptor (TRA)

    TRA takes historical prediction erros & latent representation as inputs,
    then routes the input sample to a specific predictor for training & inference.

    Args:
        input_size (int): input size (RNN/Transformer's hidden size)
        num_states (int): number of latent states (i.e., trading patterns)
            If `num_states=1`, then TRA falls back to traditional methods
        hidden_size (int): hidden size of the router
        tau (float): gumbel softmax temperature
        src_info (str): information for the router
    """

    def __init__(self, input_size, num_states=1, hidden_size=8, tau=1.0, src_info="LR_TPE"):
        super().__init__()

        assert src_info in ["LR", "TPE", "LR_TPE"], "invalid `src_info`"

        self.num_states = num_states
        self.tau = tau
        self.src_info = src_info

        self.predictors = nn.Linear(input_size, num_states)

        if self.num_states > 1:
            if "TPE" in src_info:
                self.router = nn.GRU(
                    input_size=num_states,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                )
                self.fc = nn.Linear(hidden_size + input_size if "LR" in src_info else hidden_size, num_states)
            else:
                self.fc = nn.Linear(input_size, num_states)

    def reset_parameters(self):
        for child in self.children():
            child.reset_parameters()

    def forward(self, hidden, hist_loss):

        preds = self.predictors(hidden)

        if self.num_states == 1:  # no need for router when having only one prediction
            return preds, None, None

        if "TPE" in self.src_info:
            out = self.router(hist_loss)[0][:, -1]  # TPE
            if "LR" in self.src_info:
                out = torch.cat([hidden, out], dim=-1)  # LR_TPE
        else:
            out = hidden  # LR

        out = self.fc(out)

        choice = F.gumbel_softmax(out, dim=-1, tau=self.tau, hard=True)
        prob = torch.softmax(out / self.tau, dim=-1)

        return preds, choice, prob


def evaluate(pred):
    pred = pred.rank(pct=True)  # transform into percentiles
    score = pred.score
    label = pred.label
    diff = score - label
    MSE = (diff ** 2).mean()
    MAE = (diff.abs()).mean()
    IC = score.corr(label, method="spearman")
    return {"MSE": MSE, "MAE": MAE, "IC": IC}


def average_params(params_list):
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError("the %d-th model has different params" % i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf, as_tuple=False)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


def sinkhorn(Q, n_iters=3, epsilon=0.01):
    # epsilon should be adjusted according to logits value's scale
    with torch.no_grad():
        Q = torch.exp(Q / epsilon)
        Q = shoot_infs(Q)
        for i in range(n_iters):
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= Q.sum(dim=1, keepdim=True)
    return Q


def loss_fn(pred, label):
    mask = ~torch.isnan(label)
    if len(pred.shape) == 2:
        label = label[:, None]
    return (pred[mask] - label[mask]).pow(2).mean(dim=0)


def transport_sample(all_preds, label, choice, prob, count, transport_method, training=False):
    """
    sample-wise transport

    Args:
        all_preds (torch.Tensor): predictions from all predictors, [sample x states]
        label (torch.Tensor): label, [sample]
        choice (torch.Tensor): gumbel softmax choice, [sample x states]
        prob (torch.Tensor): router predicted probility, [sample x states]
        count (list): sample counts for each day, empty list for sample-wise transport
        transport_method (str): transportation method
        training (bool): indicate training or inference
    """
    assert all_preds.shape == choice.shape
    assert len(all_preds) == len(label)
    assert transport_method in ["oracle", "router"]

    all_loss = (all_preds - label[:, None]).pow(2)  # [sample x states]
    all_loss[torch.isnan(label)] = 0.0

    if transport_method == "router":
        if training:  # router training
            pred = (all_preds * choice).sum(dim=1)  # gumbel softmax
        else:  # router inference
            pred = all_preds[range(len(all_preds)), prob.argmax(dim=-1)]  # argmax
    elif not training:  # oracle inference: always choose the model with the smallest loss
        pred = all_preds[range(len(all_preds)), all_loss.argmin(dim=-1)]
    else:  # oracle training: pred is not needed
        pred = None

    L = (all_loss - all_loss.min(dim=1, keepdim=True).values).detach()  # normalize
    P = sinkhorn(-L) if training else None  # use sinkhorn to get sample assignment during training

    if pred is not None:  # router training/inference & oracle inference loss
        loss = loss_fn(pred, label)
    else:  # oracle training loss
        loss = (all_loss * P).sum(dim=1).mean()

    return loss, pred, L, P


def transport_daily(all_preds, label, choice, prob, count, transport_method, training=False):
    """
    daily transport

    Args:
        all_preds (torch.Tensor): predictions from all predictors, [sample x states]
        label (torch.Tensor): label, [sample]
        choice (torch.Tensor): gumbel softmax choice, [days x states]
        prob (torch.Tensor): router predicted probility, [days x states]
        count (list): sample counts for each day, [days]
        transport_method (str): transportation method
        training (bool): indicate training or inference
    """
    assert len(prob) == len(count)
    assert len(all_preds) == sum(count)
    assert transport_method in ["oracle", "router"]

    all_loss = []  # loss of all predictions
    pred = []  # final predictions
    start = 0
    for i, cnt in enumerate(count):
        slc = slice(start, start + cnt)  # samples from the i-th day
        start += cnt
        tloss = loss_fn(all_preds[slc], label[slc])  # loss of the i-th day
        all_loss.append(tloss)
        if transport_method == "router":
            if training:  # router training
                tpred = all_preds[slc] @ choice[i]  # gumbel softmax
            else:  # router inference
                tpred = all_preds[slc][:, prob[i].argmax(dim=-1)]  # argmax
        elif not training:  # oracle inference: always choose the model with the smallest loss
            tpred = all_preds[slc][:, tloss.argmin(dim=-1)]
        else:  # oracle training: pred is not needed
            tpred = None
        if tpred is not None:
            pred.append(tpred)
    all_loss = torch.stack(all_loss, dim=0)  # [days x states]
    if pred:
        pred = torch.cat(pred, dim=0)  # [samples]

    L = (all_loss - all_loss.min(dim=1, keepdim=True).values).detach()  # normalize
    P = sinkhorn(-L) if training else None  # use sinkhorn to get sample assignment during training

    if len(pred):  # router training/inference & oracle inference loss
        loss = loss_fn(pred, label)
    else:  # oracle training loss
        loss = (all_loss * P).sum(dim=1).mean()

    return loss, pred, L, P

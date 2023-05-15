# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import copy
from ...utils import get_or_create_path
from ...log import get_module_logger, get_tensorboard_logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

from os.path import join
from datetime import datetime
from copy import deepcopy

from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset.handler import DataHandlerLP
from .pytorch_lstm import LSTMModel
from .pytorch_gru import GRUModel
from qlib.contrib.model.pytorch_alstm_ts import ALSTMModel


class DailyBatchSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        # calculate number of samples in each batch
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0
        self.k = 20 # the number of combined time-steps

    def __iter__(self):
        a = self.daily_index[0]
        b = self.daily_count[0]
        a_b = np.arange(a, b)
        for idx, count in zip(self.daily_index, self.daily_count):
            self.k -= 1
            if self.k >= 0: 
                missing = np.tile(a_b, self.k)
                existing = np.arange(0, idx + count)
                yield np.append(missing, existing)
            else:
                yield np.arange(-self.k*count, idx + count)

    def __len__(self):
        return len(self.data_source)


class GATs(Model):
    """GATs Model

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
        d_feat=20,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        weight_decay=0, # added by Ashot
        metric="",
        early_stop=20,
        loss="mse",
        lamb_precise_margin_ranking=0.5,
        func_precise_margin_ranking="linear", # "cubic"
        base_model="GRU",
        model_path=None,
        optimizer="adam",
        GPU=0,
        n_jobs=10,
        tensorboard_path="",
        k=20, # the number of combined time-steps
        print_iter=50,
        seed=None,
        **kwargs
    ):
        # Set logger.
        self.logger = get_module_logger("GATs")
        self.logger.info("GATs pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay # added by Ashot
        self.metric = metric
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.lamb_precise_margin_ranking = lamb_precise_margin_ranking
        self.func_precise_margin_ranking = func_precise_margin_ranking
        self.base_model = base_model
        self.model_path = model_path
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.n_jobs = n_jobs
        self.seed = seed
        self.tensorboard_path = tensorboard_path
        self.k = k
        self.print_iter = print_iter

        self.logger.info(
            "GATs parameters setting:"
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

        self.GAT_model = GATModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            base_model=self.base_model,
            k=self.k
        )
        self.logger.info("model:\n{:}".format(self.GAT_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.GAT_model)))
        
        lr_customized = [
            {"params": self.GAT_model.rnn.parameters()},
            {"params": self.GAT_model.transformation.parameters()},
            {"params": self.GAT_model.fc.parameters()},
            {"params": self.GAT_model.fc_out.parameters()},
            {"params": self.GAT_model.leaky_relu.parameters()},
            {"params": self.GAT_model.softmax.parameters()},
            {"params": self.GAT_model.a},
            {"params": self.GAT_model.alstm.parameters(), "lr": self.lr}
        ]
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.AdamW(lr_customized, lr=self.lr/10, weight_decay=self.weight_decay)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(lr_customized, lr=self.lr/10, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.GAT_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)
    
    def bce(self, pred, label):
        return F.binary_cross_entropy_with_logits(pred, label)    
    
    def margin_ranking(self, pred, label, use_mse=False):
        idx = torch.randperm(pred.size(0))
        pair_1, pair_2 = idx[::2], idx[1::2]
        if pred.size(0) % 2 == 1:
            pair_1 = pair_1[:-1]
        target = torch.sign(label[pair_1] - label[pair_2])
        loss = F.margin_ranking_loss(pred[pair_1], pred[pair_2], target, margin=0, reduction='mean')
        if use_mse:
            loss += torch.mean(torch.sqrt((pred - label) ** 2))
        
        return loss
    
    def half_margin_ranking(self, pred, label, use_mse=False):
        idx = torch.randperm(pred.size(0))
        pair_1, pair_2 = idx[::2], idx[1::2]
        if pred.size(0) % 2 == 1:
            pair_1 = pair_1[:-1]
        target = torch.sign(label[pair_1] - label[pair_2])
        pred_ord = torch.sign(pred[pair_1] - pred[pair_2])
        loss = F.margin_ranking_loss(pred[pair_1][target != pred_ord], pred[pair_2][target != pred_ord], target[target != pred_ord], margin=0.05, reduction='mean')
        if use_mse:
            loss += torch.mean(torch.sqrt((pred - label) ** 2))
        
        return loss
    
    def precise_margin_ranking(self, pred, label, use_mse=False):
        lamb = self.lamb_precise_margin_ranking
        idx = torch.randperm(pred.size(0))
        pair_1, pair_2 = idx[::2], idx[1::2]
        if pred.size(0) % 2 == 1:
            pair_1 = pair_1[:-1]

        if self.func_precise_margin_ranking == "linear":
            f = - (label[pair_1] - label[pair_2]) * (pred[pair_1] - pred[pair_2])
        elif self.func_precise_margin_ranking == "cubic":
            f = - torch.sign(label[pair_1] - label[pair_2]) * \
                torch.pow(torch.abs(label[pair_1] - label[pair_2]), 1/3) * \
                (pred[pair_1] - pred[pair_2])
        else:
            print("The func_precise_margin_ranking "
                  f"{self.func_precise_margin_ranking} is not supported!")
        loss = torch.sum(torch.maximum(torch.tensor(0).to(self.device), f))
        if use_mse:
            loss = (1 - lamb) * loss + lamb * torch.sum((pred - label) ** 2)
        return loss        
    
    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        elif self.loss == "bce":
            return self.bce(pred[mask], label[mask])
        elif self.loss == "margin_ranking":
            return self.margin_ranking(pred[mask], label[mask])
        elif self.loss == "margin_ranking_w_mse":
            return self.margin_ranking(pred[mask], label[mask], use_mse=True)
        elif self.loss == "precise_margin_ranking":
            return self.precise_margin_ranking(pred[mask], label[mask])
        elif self.loss == "precise_margin_ranking_w_mse":
            return self.precise_margin_ranking(pred[mask], label[mask], use_mse=True)
        elif self.loss == "half_margin_ranking":
            return self.half_margin_ranking(pred[mask], label[mask])
        elif self.loss == "half_margin_ranking_w_mse":
            return self.half_margin_ranking(pred[mask], label[mask], use_mse=True)

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

    def train_epoch(self, data_loader, train_loader, val_loader, epoch=0, split='train', writer=None):

        self.GAT_model.train()
        
        for batch_id, data in enumerate(data_loader):
            
            data = data.squeeze()
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)
            label = label[-(label.shape[0] // self.k):]
            pred_joint, pred_gats = self.GAT_model(feature.float())
            loss = self.loss_fn(pred_joint, label) + self.loss_fn(pred_gats, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.GAT_model.parameters(), 3.0)
            self.train_optimizer.step()
            if batch_id % self.print_iter == 0 and writer:
                train_loss, train_score = self.test_epoch(deepcopy(train_loader))
                val_loss, val_score = self.test_epoch(deepcopy(val_loader))
                writer.add_scalars(f'Loss', {'train': train_loss, 'val': val_loss}, (len(data_loader) * epoch / (data.size(0) / self.k) + batch_id) * (data.size(0) / self.k))

    def test_epoch(self, data_loader):

        self.GAT_model.eval()

        scores = []
        losses = []
        for data in data_loader:

            data = data.squeeze()
            
            feature = data[:, :, 0:-1].to(self.device)
            # feature[torch.isnan(feature)] = 0
            label = data[:, -1, -1].to(self.device)
            label = label[-(label.shape[0] // self.k):]
            pred_joint, pred_gats = self.GAT_model(feature.float())
            loss = self.loss_fn(pred_joint, label) + self.loss_fn(pred_gats, label)
            losses.append(loss.item())

            score = self.metric_fn(pred_joint, label)
            scores.append(score.item())
        
        self.GAT_model.train()
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

        train_loader = DataLoader(dl_train, sampler=sampler_train, num_workers=self.n_jobs, drop_last=True)
        valid_loader = DataLoader(dl_valid, sampler=sampler_valid, num_workers=self.n_jobs, drop_last=True)

        save_path = get_or_create_path(save_path)
   
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        tboard_writer = get_tensorboard_logger(save_path=join(self.tensorboard_path, f"GATs_ALSTM_{current_time}"))
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
        
        # --- pretrained gats_decay_001 ---
        basic_gats_path = ("/home/ashotnanyan/qlib/examples/test_gats/decay_001/1/"
                           "69bdd9a5a84c48e3a1852e76809315e1/artifacts/params_torch.pkl")
        pretrained_gats = torch.load(basic_gats_path, map_location=self.device)
        pretrained_gats_dict = pretrained_gats.GAT_model.state_dict()
        # ---------------------------------

        if self.model_path is not None:
            self.logger.info("Loading pretrained model...")
            pretrained_model.load_state_dict(torch.load(self.model_path, map_location=self.device))

        model_dict = self.GAT_model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_model.state_dict().items() if k in model_dict  # pylint: disable=E1135
        }
        model_dict.update(pretrained_dict)
        model_dict.update(pretrained_gats_dict) # added by Ashot
        self.GAT_model.load_state_dict(model_dict)
        self.logger.info("Loading pretrained model Done...")

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(deepcopy(train_loader), deepcopy(train_loader), valid_loader, epoch=step, split='train', writer=tboard_writer)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(deepcopy(train_loader))
            val_loss, val_score = self.test_epoch(deepcopy(valid_loader))
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.GAT_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.GAT_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        sampler_test = DailyBatchSampler(dl_test)
        test_loader = DataLoader(dl_test, sampler=sampler_test, num_workers=self.n_jobs)
        self.GAT_model.eval()
        preds = []

        for data in test_loader:

            data = data.squeeze()
            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                pred, _ = self.GAT_model(feature.float())
                pred = pred.detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())


class GATModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU", k=20):
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

        self.k = k # the number of combined time-steps

        self.hidden_size = hidden_size
        self.d_feat = d_feat
        self.alstm = ALSTMModel(d_feat=self.hidden_size, 
                                hidden_size=64, 
                                num_layers=2, 
                                dropout=0.8, 
                                rnn_type="GRU")
        self.transformation = nn.Linear(self.hidden_size, self.hidden_size)
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
        self.a.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=2)       

    def cal_attention(self, x, y):
        x = self.transformation(x)
        y = self.transformation(y)
        sample_num = x.shape[1]
        dim = x.shape[2]
        e_x = x.unsqueeze(dim=1)
        e_x = e_x.expand(-1, sample_num, -1, -1)
        e_y = torch.transpose(e_x, 1, 2)
        attention_in = torch.cat((e_x, e_y), 3).view(-1, dim * 2)
        self.a_t = torch.t(self.a)
        attention_out = self.a_t.mm(torch.t(attention_in)).view(-1, sample_num, sample_num)
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out) # 20x300x300
        return att_weight

    def forward(self, x):
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]
        _, hidden_size = hidden.size()
        hidden = hidden.view(self.k, -1, hidden_size)
        att_weight = self.cal_attention(hidden, hidden)
        hidden = att_weight.matmul(hidden) + hidden
        hidden = self.fc(hidden)
        hidden = hidden.transpose(1, 0)
        alstm_out = self.alstm(hidden)
        
        hidden = hidden[:, -1, :]
        hidden = self.leaky_relu(hidden)
        return alstm_out, self.fc_out(hidden).squeeze()
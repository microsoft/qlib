from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
from typing import Text, Union
import urllib.request
import copy
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import datetime
from qlib.contrib.model.pytorch_utils import count_parameters
from qlib.data.dataset import DatasetH
from qlib.model.base import Model
from qlib.utils import get_or_create_path
from qlib.log import get_module_logger
from qlib.contrib.model.pytorch_lstm import LSTMModel
from qlib.contrib.model.pytorch_gru import GRUModel
from qlib.data.dataset.handler import DataHandlerLP


class MTMD(Model):
    """MTMD Model

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
        base_model="LSTM",
        model_path=None,
        stock2concept=None,
        stock_index=None,
        optimizer="adam",
        GPU=0,
        seed=None,
        K=3,
        **kwargs
    ):
        # Set logger.
        self.logger = get_module_logger("MTMD")
        self.logger.info("MTMD pytorch version...")

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
        self.stock2concept = stock2concept
        self.stock_index = stock_index
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.K=K

        self.logger.info(
            "MTMD parameters setting:"
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
            "\nstock2concept : {}"
            "\nstock_index : {}"
            "\nuse_GPU : {}"
            "\nseed : {}"
            "\nK : {}".format(
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
                stock2concept,
                stock_index,
                GPU,
                seed,
                K,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.MTMD_model = MTMDModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            base_model=self.base_model,
            K=self.K,
        )
        self.logger.info("model:\n{:}".format(self.MTMD_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.MTMD_model)))
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.MTMD_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.MTMD_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.MTMD_model.to(self.device)

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

    def train_epoch(self, x_train, y_train, stock_index,m_items=None):
        stock2concept_matrix = np.load(self.stock2concept)
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)
        stock_index = stock_index.values
        stock_index[np.isnan(stock_index)] = 733
        self.MTMD_model.train()

        # organize the train data into daily batches
        daily_index, daily_count = self.get_daily_inter(x_train, shuffle=True)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_train_values[batch]).float().to(self.device)
            concept_matrix = torch.from_numpy(stock2concept_matrix[stock_index[batch]]).float().to(self.device)
            label = torch.from_numpy(y_train_values[batch]).float().to(self.device)

            pred,m_items = self.MTMD_model(feature, concept_matrix,m_items,train=True)
    
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.MTMD_model.parameters(), 3.0)
            self.train_optimizer.step()
        return m_items

    def test_epoch(self, data_x, data_y, stock_index,m_items=None,train=False):
        # prepare training data
        stock2concept_matrix = np.load(self.stock2concept)
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)
        stock_index = stock_index.values
        stock_index[np.isnan(stock_index)] = 733
        self.MTMD_model.eval()

        scores = []
        losses = []

        # organize the test data into daily batches
        daily_index, daily_count = self.get_daily_inter(data_x, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)
            concept_matrix = torch.from_numpy(stock2concept_matrix[stock_index[batch]]).float().to(self.device)
            label = torch.from_numpy(y_values[batch]).float().to(self.device)
            with torch.no_grad():
      
                pred,m_items = self.MTMD_model(feature, concept_matrix,m_items,train=False)
        
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

        if not os.path.exists(self.stock2concept):
            url = "http://fintech.msra.cn/stock_data/downloads/qlib_csi300_stock2concept.npy"
            urllib.request.urlretrieve(url, self.stock2concept)

        stock_index = np.load(self.stock_index, allow_pickle=True).item()
        df_train["stock_index"] = 733
        df_train["stock_index"] = df_train.index.get_level_values("instrument").map(stock_index)
        df_valid["stock_index"] = 733
        df_valid["stock_index"] = df_valid.index.get_level_values("instrument").map(stock_index)

        x_train, y_train, stock_index_train = df_train["feature"], df_train["label"], df_train["stock_index"]
        x_valid, y_valid, stock_index_valid = df_valid["feature"], df_valid["label"], df_valid["stock_index"]

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []
        m_item0 = F.normalize(torch.rand((96,64), dtype=torch.float), dim=1).cuda()
        m_item1 = F.normalize(torch.rand((96,64), dtype=torch.float), dim=1).cuda()
        m_items = [m_item0, m_item1]
        # load pretrained base_model
        if self.base_model == "LSTM":
            pretrained_model = LSTMModel()
        elif self.base_model == "GRU":
            pretrained_model = GRUModel()
        else:
            raise ValueError("unknown base model name `%s`" % self.base_model)

        if self.model_path is not None:
            self.logger.info("Loading pretrained model...")
            pretrained_model.load_state_dict(torch.load(self.model_path))

        model_dict = self.MTMD_model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_model.state_dict().items() if k in model_dict  # pylint: disable=E1135
        }
        model_dict.update(pretrained_dict)
        self.MTMD_model.load_state_dict(model_dict)
        self.logger.info("Loading pretrained model Done...")

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(x_train, y_train, stock_index_train,m_items)

            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(x_train, y_train, stock_index_train,m_items)
            val_loss, val_score = self.test_epoch(x_valid, y_valid, stock_index_valid,m_items)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.MTMD_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.MTMD_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        stock2concept_matrix = np.load(self.stock2concept)
        stock_index = np.load(self.stock_index, allow_pickle=True).item()
        df_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        df_test["stock_index"] = 733
        df_test["stock_index"] = df_test.index.get_level_values("instrument").map(stock_index)
        stock_index_test = df_test["stock_index"].values
        stock_index_test[np.isnan(stock_index_test)] = 733
        stock_index_test = stock_index_test.astype("int")
        df_test = df_test.drop(["stock_index"], axis=1)
        index = df_test.index
        m_item0 = F.normalize(torch.rand((96,64), dtype=torch.float), dim=1).cuda()
        m_item1 = F.normalize(torch.rand((96,64), dtype=torch.float), dim=1).cuda()
        m_items=m_item0, m_item1
        train=False

        self.MTMD_model.eval()
        x_values = df_test.values
        preds = []
        

        # organize the data into daily batches
        daily_index, daily_count = self.get_daily_inter(df_test, shuffle=False)
        


        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            x_batch = torch.from_numpy(x_values[batch]).float().to(self.device)
            concept_matrix = torch.from_numpy(stock2concept_matrix[stock_index_test[batch]]).float().to(self.device)

            with torch.no_grad():
                pred,m_items = self.MTMD_model(x_batch, concept_matrix,m_items,train)
            pred=pred.detach().cpu().numpy()
            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)

def cal_cos_similarity(x, y): # the 2nd dimension of x and y are the same
    xy = x.mm(torch.t(y))
    x_norm = torch.sqrt(torch.sum(x*x, dim =1)).reshape(-1, 1)
    y_norm = torch.sqrt(torch.sum(y*y, dim =1)).reshape(-1, 1)
    cos_similarity = xy/x_norm.mm(torch.t(y_norm))
    cos_similarity[cos_similarity != cos_similarity] = 0
    return cos_similarity
def get_score(K, Q):
    score = torch.matmul(Q,torch.t(K) )
    score_query = F.softmax(score, dim=0)
    score_memory = F.softmax(score,dim=1)
    return score_query, score_memory

def read(Q,K):
    softmax_score_query, softmax_score_memory = get_score(K,Q)
    query_reshape = Q.contiguous()
    concat_memory = torch.matmul(softmax_score_memory.detach(), K) # (b X h X w) X d
    updated_query = query_reshape*concat_memory
    return updated_query,softmax_score_query,softmax_score_memory

def upload(Q,K):
    softmax_score_query, softmax_score_memory = get_score(K,Q)
    query_reshape = Q.contiguous()
    _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
    _, updating_indices = torch.topk(softmax_score_query, 1, dim=0)
    m, d = K.size()
    query_update = torch.zeros((m,d)).cuda()
    random_update = torch.zeros((m,d)).cuda()
    for i in range(m):
        idx = torch.nonzero(gathering_indices.squeeze(1)==i)
        a, _ = idx.size()
        if a != 0:
            query_update[i] = torch.sum(((softmax_score_query[idx,i] / torch.max(softmax_score_query[:,i])) 
                                        *query_reshape[idx].squeeze(1)), dim=0)
        else:
            query_update[i] = 0 
    updated_memory = F.normalize(query_update.cuda() + K.cuda(), dim=1)
    return updated_memory.detach()


class MTMDModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2,K=3, dropout=0.0, base_model="GRU"):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.K = K

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

        self.fc_ps = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_ps.weight)
        self.fc_hs = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_hs.weight)

        self.fc_ps_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_ps_fore.weight)
        self.fc_hs_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_hs_fore.weight)

        self.fc_ps_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_ps_back.weight)
        self.fc_hs_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_hs_back.weight)
        self.fc_indi = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_indi.weight)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax_s2t = torch.nn.Softmax(dim = 0)
        self.softmax_t2s = torch.nn.Softmax(dim = 1)
        
        self.fc_out_ps = nn.Linear(hidden_size, 1)
        self.fc_out_hs = nn.Linear(hidden_size, 1)
        self.fc_out_indi = nn.Linear(hidden_size, 1)
        self.fc_out = nn.Linear(hidden_size, 1)
        
        
    def cal_cos_similarity(self, x, y): # the 2nd dimension of x and y are the same
        xy = x.mm(torch.t(y))
        x_norm = torch.sqrt(torch.sum(x*x, dim =1)).reshape(-1, 1)
        y_norm = torch.sqrt(torch.sum(y*y, dim =1)).reshape(-1, 1)
        cos_similarity = xy/x_norm.mm(torch.t(y_norm))
        cos_similarity[cos_similarity != cos_similarity] = 0
        return cos_similarity
    def forward(self, x, concept_matrix,m_items,train):
        device = torch.device(torch.get_device(x))
        m_item0, m_item1 = m_items
        x_hidden = x.reshape(len(x), self.d_feat, -1) # [N, F, T]      
        x_hidden = x_hidden.permute(0, 2, 1) # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden)
        x_hidden = x_hidden[:, -1, :]
        # Predefined Concept Module
        # representation initialization
        stock_to_concept = concept_matrix

        stock_to_concept_sum = torch.sum(stock_to_concept, 0).reshape(1, -1).repeat(stock_to_concept.shape[0], 1)
        stock_to_concept_sum = stock_to_concept_sum.mul(concept_matrix)

        stock_to_concept_sum = stock_to_concept_sum + (torch.ones(stock_to_concept.shape[0], stock_to_concept.shape[1]).to(device))
        stock_to_concept = stock_to_concept / stock_to_concept_sum
        hidden = torch.t(stock_to_concept).mm(x_hidden)
        
        hidden = hidden[hidden.sum(1)!=0]
            
        stock_to_concept = x_hidden.mm(torch.t(hidden))
        # stock_to_concept = cal_cos_similarity(x_hidden, hidden)
        
        stock_to_concept = self.softmax_s2t(stock_to_concept)
        hidden = torch.t(stock_to_concept).mm(x_hidden)

        concept_to_stock = cal_cos_similarity(x_hidden, hidden)
        concept_to_stock = self.softmax_t2s(concept_to_stock)

        p_shared_info = concept_to_stock.mm(hidden)
        p_shared_info = self.fc_ps(p_shared_info)

        p_shared_info, _,softmax_score_memory0 = read(p_shared_info,m_item0)
        if train:
            softmax_score_memory0 = upload(p_shared_info,m_item0)
        else:
            softmax_score_memory0 = m_item0

        p_shared_back = self.fc_ps_back(p_shared_info) # output of backcast branch
        output_ps = self.fc_ps_fore(p_shared_info) # output of forecast branch
        output_ps = self.leaky_relu(output_ps)

        pred_ps = self.fc_out_ps(output_ps).squeeze()
        
        # Hidden Concept Module
        h_shared_info = x_hidden - p_shared_back
        hidden = h_shared_info        
        h_stock_to_concept = cal_cos_similarity(hidden, hidden)
        
        dim = h_stock_to_concept.shape[0]
        diag = h_stock_to_concept.diagonal(0)
        h_stock_to_concept = h_stock_to_concept * (torch.ones(dim, dim) - torch.eye(dim)).to(device)
        row = torch.linspace(0, dim-1, dim).reshape([-1, 1]).repeat(1, self.K).reshape(1, -1).long().to(device)
        column = torch.topk(h_stock_to_concept, self.K, dim = 1)[1].reshape(1, -1)
        mask = torch.zeros([h_stock_to_concept.shape[0], h_stock_to_concept.shape[1]], device = h_stock_to_concept.device)
        mask[row, column] = 1
        h_stock_to_concept = h_stock_to_concept * mask
        h_stock_to_concept = h_stock_to_concept + torch.diag_embed((h_stock_to_concept.sum(0)!=0).float()*diag)
        hidden = torch.t(h_shared_info).mm(h_stock_to_concept).t()
        hidden = hidden[hidden.sum(1)!=0]
        
        h_concept_to_stock = cal_cos_similarity(h_shared_info, hidden)
        h_concept_to_stock = self.softmax_t2s(h_concept_to_stock)
        h_shared_info = h_concept_to_stock.mm(hidden)
        h_shared_info = self.fc_hs(h_shared_info)

        h_shared_info, _,softmax_score_memory1 = read(h_shared_info,m_item1)
        
        if train:
            softmax_score_memory1 = upload(h_shared_info,m_item1)
        else:
            softmax_score_memory1 = m_item1
   
        h_shared_back = self.fc_hs_back(h_shared_info)
        output_hs = self.fc_hs_fore(h_shared_info)
        output_hs = self.leaky_relu(output_hs)
        #pred_hs = self.fc_out_hs(output_hs).squeeze()
        # Individual Information Module
        individual_info  = x_hidden - p_shared_back - h_shared_back
        output_indi = individual_info
        output_indi = self.fc_indi(output_indi)
        output_indi = self.leaky_relu(output_indi)
        #pred_indi = self.fc_out_indi(output_indi).squeeze()
        # Stock Trend Prediction
        all_info = output_ps + output_hs + output_indi
        pred_all = self.fc_out(all_info).squeeze()

        return pred_all, [softmax_score_memory0,softmax_score_memory1]
 

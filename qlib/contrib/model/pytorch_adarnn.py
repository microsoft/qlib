# Copyright (c) Microsoft Corporation.
import os
from torch.utils.data import Dataset, DataLoader

import copy
from typing import Text, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from qlib.contrib.model.pytorch_utils import count_parameters
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import get_module_logger
from qlib.model.base import Model
from qlib.utils import get_or_create_path


class ADARNN(Model):
    """ADARNN Model

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
        pre_epoch=40,
        dw=0.5,
        loss_type="cosine",
        len_seq=60,
        len_win=0,
        lr=0.001,
        metric="mse",
        batch_size=2000,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        n_splits=2,
        GPU=0,
        seed=None,
        **kwargs
    ):
        # Set logger.
        self.logger = get_module_logger("ADARNN")
        self.logger.info("ADARNN pytorch version...")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)

        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.pre_epoch = pre_epoch
        self.dw = dw
        self.loss_type = loss_type
        self.len_seq = len_seq
        self.len_win = len_win
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.n_splits = n_splits
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed

        self.logger.info(
            "ADARNN parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
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
                batch_size,
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

        n_hiddens = [hidden_size for _ in range(num_layers)]
        self.model = AdaRNN(
            use_bottleneck=False,
            bottleneck_width=64,
            n_input=d_feat,
            n_hiddens=n_hiddens,
            n_output=1,
            dropout=dropout,
            model_type="AdaRNN",
            len_seq=len_seq,
            trans_loss=loss_type,
        )
        self.logger.info("model:\n{:}".format(self.model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def train_AdaRNN(self, train_loader_list, epoch, dist_old=None, weight_mat=None):
        self.model.train()
        criterion = nn.MSELoss()
        dist_mat = torch.zeros(self.num_layers, self.len_seq).to(self.device)
        len_loader = np.inf
        for loader in train_loader_list:
            if len(loader) < len_loader:
                len_loader = len(loader)
        for data_all in zip(*train_loader_list):
            #  for data_all in zip(*train_loader_list):
            self.train_optimizer.zero_grad()
            list_feat = []
            list_label = []
            for data in data_all:
                # feature :[36, 24, 6]
                feature, label_reg = data[0].to(self.device).float(), data[1].to(self.device).float()
                list_feat.append(feature)
                list_label.append(label_reg)
            flag = False
            index = get_index(len(data_all) - 1)
            for temp_index in index:
                s1 = temp_index[0]
                s2 = temp_index[1]
                if list_feat[s1].shape[0] != list_feat[s2].shape[0]:
                    flag = True
                    break
            if flag:
                continue

            total_loss = torch.zeros(1).to(self.device)
            for i, n in enumerate(index):
                feature_s = list_feat[n[0]]
                feature_t = list_feat[n[1]]
                label_reg_s = list_label[n[0]]
                label_reg_t = list_label[n[1]]
                feature_all = torch.cat((feature_s, feature_t), 0)

                if epoch < self.pre_epoch:
                    pred_all, loss_transfer, out_weight_list = self.model.forward_pre_train(
                        feature_all, len_win=self.len_win
                    )
                else:
                    pred_all, loss_transfer, dist, weight_mat = self.model.forward_Boosting(feature_all, weight_mat)
                    dist_mat = dist_mat + dist
                pred_s = pred_all[0 : feature_s.size(0)]
                pred_t = pred_all[feature_s.size(0) :]

                loss_s = criterion(pred_s, label_reg_s)
                loss_t = criterion(pred_t, label_reg_t)

                total_loss = total_loss + loss_s + loss_t + self.dw * loss_transfer
            self.train_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()
        if epoch >= self.pre_epoch:
            if epoch > self.pre_epoch:
                weight_mat = self.model.update_weight_Boosting(weight_mat, dist_old, dist_mat)
            return weight_mat, dist_mat
        else:
            weight_mat = self.transform_type(out_weight_list)
            return weight_mat, None

    def calc_all_metrics(self, pred):
        """pred is a pandas dataframe that has two attributes: score (pred) and label (real)"""
        res = {}
        ic = pred.groupby(level="datetime").apply(lambda x: x.label.corr(x.score))
        rank_ic = pred.groupby(level="datetime").apply(lambda x: x.label.corr(x.score, method="spearman"))
        res["ic"] = ic.mean()
        res["icir"] = ic.mean() / ic.std()
        res["ric"] = rank_ic.mean()
        res["ricir"] = rank_ic.mean() / rank_ic.std()
        res["mse"] = -(pred["label"] - pred["score"]).mean()
        res["loss"] = res["mse"]
        return res

    def test_epoch(self, df):
        self.model.eval()
        preds = self.infer(df["feature"])
        label = df["label"].squeeze()
        preds = pd.DataFrame({"label": label, "score": preds}, index=df.index)
        metrics = self.calc_all_metrics(preds)
        return metrics

    def log_metrics(self, mode, metrics):
        metrics = ["{}/{}: {:.6f}".format(k, mode, v) for k, v in metrics.items()]
        metrics = ", ".join(metrics)
        self.logger.info(metrics)

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
        #  splits = ['2011-06-30']
        days = df_train.index.get_level_values(level=0).unique()
        train_splits = np.array_split(days, self.n_splits)
        train_splits = [df_train[s[0] : s[-1]] for s in train_splits]
        train_loader_list = [get_stock_loader(df, self.batch_size) for df in train_splits]

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True
        best_score = -np.inf
        best_epoch = 0
        weight_mat, dist_mat = None, None

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            weight_mat, dist_mat = self.train_AdaRNN(train_loader_list, step, dist_mat, weight_mat)
            self.logger.info("evaluating...")
            train_metrics = self.test_epoch(df_train)
            valid_metrics = self.test_epoch(df_valid)
            self.log_metrics("train: ", train_metrics)
            self.log_metrics("valid: ", valid_metrics)

            valid_score = valid_metrics[self.metric]
            train_score = train_metrics[self.metric]
            evals_result["train"].append(train_score)
            evals_result["valid"].append(valid_score)
            if valid_score > best_score:
                best_score = valid_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()
        return best_score

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return self.infer(x_test)

    def infer(self, x_test):
        index = x_test.index
        self.model.eval()
        x_values = x_test.values
        sample_num = x_values.shape[0]
        x_values = x_values.reshape(sample_num, self.d_feat, -1).transpose(0, 2, 1)
        preds = []

        for begin in range(sample_num)[:: self.batch_size]:

            if sample_num - begin < self.batch_size:
                end = sample_num
            else:
                end = begin + self.batch_size

            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)

            with torch.no_grad():
                pred = self.model.predict(x_batch).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)

    def transform_type(self, init_weight):
        weight = torch.ones(self.num_layers, self.len_seq).to(self.device)
        for i in range(self.num_layers):
            for j in range(self.len_seq):
                weight[i, j] = init_weight[i][j].item()
        return weight


class data_loader(Dataset):
    def __init__(self, df):
        self.df_feature = df["feature"]
        self.df_label_reg = df["label"]
        self.df_index = df.index
        self.df_feature = torch.tensor(
            self.df_feature.values.reshape(-1, 6, 60).transpose(0, 2, 1), dtype=torch.float32
        )
        self.df_label_reg = torch.tensor(self.df_label_reg.values.reshape(-1), dtype=torch.float32)

    def __getitem__(self, index):
        sample, label_reg = self.df_feature[index], self.df_label_reg[index]
        return sample, label_reg

    def __len__(self):
        return len(self.df_feature)


def get_stock_loader(df, batch_size, shuffle=True):
    train_loader = DataLoader(data_loader(df), batch_size=batch_size, shuffle=shuffle)
    return train_loader


def get_index(num_domain=2):
    index = []
    for i in range(num_domain):
        for j in range(i + 1, num_domain + 1):
            index.append((i, j))
    return index


class AdaRNN(nn.Module):
    """
    model_type:  'Boosting', 'AdaRNN'
    """

    def __init__(
        self,
        use_bottleneck=False,
        bottleneck_width=256,
        n_input=128,
        n_hiddens=[64, 64],
        n_output=6,
        dropout=0.0,
        len_seq=9,
        model_type="AdaRNN",
        trans_loss="mmd",
        GPU=0,
    ):
        super(AdaRNN, self).__init__()
        self.use_bottleneck = use_bottleneck
        self.n_input = n_input
        self.num_layers = len(n_hiddens)
        self.hiddens = n_hiddens
        self.n_output = n_output
        self.model_type = model_type
        self.trans_loss = trans_loss
        self.len_seq = len_seq
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        in_size = self.n_input

        features = nn.ModuleList()
        for hidden in n_hiddens:
            rnn = nn.GRU(input_size=in_size, num_layers=1, hidden_size=hidden, batch_first=True, dropout=dropout)
            features.append(rnn)
            in_size = hidden
        self.features = nn.Sequential(*features)

        if use_bottleneck is True:  # finance
            self.bottleneck = nn.Sequential(
                nn.Linear(n_hiddens[-1], bottleneck_width),
                nn.Linear(bottleneck_width, bottleneck_width),
                nn.BatchNorm1d(bottleneck_width),
                nn.ReLU(),
                nn.Dropout(),
            )
            self.bottleneck[0].weight.data.normal_(0, 0.005)
            self.bottleneck[0].bias.data.fill_(0.1)
            self.bottleneck[1].weight.data.normal_(0, 0.005)
            self.bottleneck[1].bias.data.fill_(0.1)
            self.fc = nn.Linear(bottleneck_width, n_output)
            torch.nn.init.xavier_normal_(self.fc.weight)
        else:
            self.fc_out = nn.Linear(n_hiddens[-1], self.n_output)

        if self.model_type == "AdaRNN":
            gate = nn.ModuleList()
            for i in range(len(n_hiddens)):
                gate_weight = nn.Linear(len_seq * self.hiddens[i] * 2, len_seq)
                gate.append(gate_weight)
            self.gate = gate

            bnlst = nn.ModuleList()
            for i in range(len(n_hiddens)):
                bnlst.append(nn.BatchNorm1d(len_seq))
            self.bn_lst = bnlst
            self.softmax = torch.nn.Softmax(dim=0)
            self.init_layers()

    def init_layers(self):
        for i in range(len(self.hiddens)):
            self.gate[i].weight.data.normal_(0, 0.05)
            self.gate[i].bias.data.fill_(0.0)

    def forward_pre_train(self, x, len_win=0):
        out = self.gru_features(x)
        fea = out[0]  # [2N,L,H]
        if self.use_bottleneck is True:
            fea_bottleneck = self.bottleneck(fea[:, -1, :])
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(fea[:, -1, :]).squeeze()  # [N,]

        out_list_all, out_weight_list = out[1], out[2]
        out_list_s, out_list_t = self.get_features(out_list_all)
        loss_transfer = torch.zeros((1,)).to(self.device)
        for i, n in enumerate(out_list_s):
            criterion_transder = TransferLoss(loss_type=self.trans_loss, input_dim=n.shape[2])
            h_start = 0
            for j in range(h_start, self.len_seq, 1):
                i_start = j - len_win if j - len_win >= 0 else 0
                i_end = j + len_win if j + len_win < self.len_seq else self.len_seq - 1
                for k in range(i_start, i_end + 1):
                    weight = (
                        out_weight_list[i][j]
                        if self.model_type == "AdaRNN"
                        else 1 / (self.len_seq - h_start) * (2 * len_win + 1)
                    )
                    loss_transfer = loss_transfer + weight * criterion_transder.compute(
                        n[:, j, :], out_list_t[i][:, k, :]
                    )
        return fc_out, loss_transfer, out_weight_list

    def gru_features(self, x, predict=False):
        x_input = x
        out = None
        out_lis = []
        out_weight_list = [] if (self.model_type == "AdaRNN") else None
        for i in range(self.num_layers):
            out, _ = self.features[i](x_input.float())
            x_input = out
            out_lis.append(out)
            if self.model_type == "AdaRNN" and predict is False:
                out_gate = self.process_gate_weight(x_input, i)
                out_weight_list.append(out_gate)
        return out, out_lis, out_weight_list

    def process_gate_weight(self, out, index):
        x_s = out[0 : int(out.shape[0] // 2)]
        x_t = out[out.shape[0] // 2 : out.shape[0]]
        x_all = torch.cat((x_s, x_t), 2)
        x_all = x_all.view(x_all.shape[0], -1)
        weight = torch.sigmoid(self.bn_lst[index](self.gate[index](x_all.float())))
        weight = torch.mean(weight, dim=0)
        res = self.softmax(weight).squeeze()
        return res

    def get_features(self, output_list):
        fea_list_src, fea_list_tar = [], []
        for fea in output_list:
            fea_list_src.append(fea[0 : fea.size(0) // 2])
            fea_list_tar.append(fea[fea.size(0) // 2 :])
        return fea_list_src, fea_list_tar

    # For Boosting-based
    def forward_Boosting(self, x, weight_mat=None):
        out = self.gru_features(x)
        fea = out[0]
        if self.use_bottleneck:
            fea_bottleneck = self.bottleneck(fea[:, -1, :])
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(fea[:, -1, :]).squeeze()

        out_list_all = out[1]
        out_list_s, out_list_t = self.get_features(out_list_all)
        loss_transfer = torch.zeros((1,)).to(self.device)
        if weight_mat is None:
            weight = (1.0 / self.len_seq * torch.ones(self.num_layers, self.len_seq)).to(self.device)
        else:
            weight = weight_mat
        dist_mat = torch.zeros(self.num_layers, self.len_seq).to(self.device)
        for i, n in enumerate(out_list_s):
            criterion_transder = TransferLoss(loss_type=self.trans_loss, input_dim=n.shape[2])
            for j in range(self.len_seq):
                loss_trans = criterion_transder.compute(n[:, j, :], out_list_t[i][:, j, :])
                loss_transfer = loss_transfer + weight[i, j] * loss_trans
                dist_mat[i, j] = loss_trans
        return fc_out, loss_transfer, dist_mat, weight

    # For Boosting-based
    def update_weight_Boosting(self, weight_mat, dist_old, dist_new):
        epsilon = 1e-5
        dist_old = dist_old.detach()
        dist_new = dist_new.detach()
        ind = dist_new > dist_old + epsilon
        weight_mat[ind] = weight_mat[ind] * (1 + torch.sigmoid(dist_new[ind] - dist_old[ind]))
        weight_norm = torch.norm(weight_mat, dim=1, p=1)
        weight_mat = weight_mat / weight_norm.t().unsqueeze(1).repeat(1, self.len_seq)
        return weight_mat

    def predict(self, x):
        out = self.gru_features(x, predict=True)
        fea = out[0]
        if self.use_bottleneck is True:
            fea_bottleneck = self.bottleneck(fea[:, -1, :])
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(fea[:, -1, :]).squeeze()
        return fc_out


class TransferLoss:
    def __init__(self, loss_type="cosine", input_dim=512, GPU=0):
        """
        Supported loss_type: mmd(mmd_lin), mmd_rbf, coral, cosine, kl, js, mine, adv
        """
        self.loss_type = loss_type
        self.input_dim = input_dim
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")

    def compute(self, X, Y):
        """Compute adaptation loss

        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix

        Returns:
            [tensor] -- transfer loss
        """
        if self.loss_type in ("mmd_lin", "mmd"):
            mmdloss = MMD_loss(kernel_type="linear")
            loss = mmdloss(X, Y)
        elif self.loss_type == "coral":
            loss = CORAL(X, Y, self.device)
        elif self.loss_type in ("cosine", "cos"):
            loss = 1 - cosine(X, Y)
        elif self.loss_type == "kl":
            loss = kl_div(X, Y)
        elif self.loss_type == "js":
            loss = js(X, Y)
        elif self.loss_type == "mine":
            mine_model = Mine_estimator(input_dim=self.input_dim, hidden_dim=60).to(self.device)
            loss = mine_model(X, Y)
        elif self.loss_type == "adv":
            loss = adv(X, Y, self.device, input_dim=self.input_dim, hidden_dim=32)
        elif self.loss_type == "mmd_rbf":
            mmdloss = MMD_loss(kernel_type="rbf")
            loss = mmdloss(X, Y)
        elif self.loss_type == "pairwise":
            pair_mat = pairwise_dist(X, Y)
            loss = torch.norm(pair_mat)

        return loss


def cosine(source, target):
    source, target = source.mean(), target.mean()
    cos = nn.CosineSimilarity(dim=0)
    loss = cos(source, target)
    return loss.mean()


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dis1 = nn.Linear(input_dim, hidden_dim)
        self.dis2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.dis1(x))
        x = self.dis2(x)
        x = torch.sigmoid(x)
        return x


def adv(source, target, device, input_dim=256, hidden_dim=512):
    domain_loss = nn.BCELoss()
    # !!! Pay attention to .cuda !!!
    adv_net = Discriminator(input_dim, hidden_dim).to(device)
    domain_src = torch.ones(len(source)).to(device)
    domain_tar = torch.zeros(len(target)).to(device)
    domain_src, domain_tar = domain_src.view(domain_src.shape[0], 1), domain_tar.view(domain_tar.shape[0], 1)
    reverse_src = ReverseLayerF.apply(source, 1)
    reverse_tar = ReverseLayerF.apply(target, 1)
    pred_src = adv_net(reverse_src)
    pred_tar = adv_net(reverse_tar)
    loss_s, loss_t = domain_loss(pred_src, domain_src), domain_loss(pred_tar, domain_tar)
    loss = loss_s + loss_t
    return loss


def CORAL(source, target, device):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # source covariance
    tmp_s = torch.ones((1, ns)).to(device) @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)).to(device) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - ct).pow(2).sum()
    loss = loss / (4 * d * d)

    return loss


class MMD_loss(nn.Module):
    def __init__(self, kernel_type="linear", kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd(self, X, Y):
        delta = X.mean(axis=0) - Y.mean(axis=0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == "linear":
            return self.linear_mmd(source, target)
        elif self.kernel_type == "rbf":
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma
            )
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            return loss


class Mine_estimator(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512):
        super(Mine_estimator, self).__init__()
        self.mine_model = Mine(input_dim, hidden_dim)

    def forward(self, X, Y):
        Y_shffle = Y[torch.randperm(len(Y))]
        loss_joint = self.mine_model(X, Y)
        loss_marginal = self.mine_model(X, Y_shffle)
        ret = torch.mean(loss_joint) - torch.log(torch.mean(torch.exp(loss_marginal)))
        loss = -ret
        return loss


class Mine(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512):
        super(Mine, self).__init__()
        self.fc1_x = nn.Linear(input_dim, hidden_dim)
        self.fc1_y = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, y):
        h1 = F.leaky_relu(self.fc1_x(x) + self.fc1_y(y))
        h2 = self.fc2(h1)
        return h2


def pairwise_dist(X, Y):
    n, d = X.shape
    m, _ = Y.shape
    assert d == Y.shape[1]
    a = X.unsqueeze(1).expand(n, m, d)
    b = Y.unsqueeze(0).expand(n, m, d)
    return torch.pow(a - b, 2).sum(2)


def pairwise_dist_np(X, Y):
    n, d = X.shape
    m, _ = Y.shape
    assert d == Y.shape[1]
    a = np.expand_dims(X, 1)
    b = np.expand_dims(Y, 0)
    a = np.tile(a, (1, m, 1))
    b = np.tile(b, (n, 1, 1))
    return np.power(a - b, 2).sum(2)


def pa(X, Y):
    XY = np.dot(X, Y.T)
    XX = np.sum(np.square(X), axis=1)
    XX = np.transpose([XX])
    YY = np.sum(np.square(Y), axis=1)
    dist = XX + YY - 2 * XY

    return dist


def kl_div(source, target):
    if len(source) < len(target):
        target = target[: len(source)]
    elif len(source) > len(target):
        source = source[: len(target)]
    criterion = nn.KLDivLoss(reduction="batchmean")
    loss = criterion(source.log(), target)
    return loss


def js(source, target):
    if len(source) < len(target):
        target = target[: len(source)]
    elif len(source) > len(target):
        source = source[: len(target)]
    M = 0.5 * (source + target)
    loss_1, loss_2 = kl_div(source, M), kl_div(target, M)
    return 0.5 * (loss_1 + loss_2)

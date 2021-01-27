import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
import sys

from tianshou.data import to_torch


class Attention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.get_w = nn.Sequential(nn.Linear(in_dim * 2, in_dim), nn.ReLU(), nn.Linear(in_dim, 1))

        self.fc = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),)

    def forward(self, value, key):
        key = key.unsqueeze(dim=1)
        length = value.shape[1]
        key = key.repeat([1, length, 1])
        weight = self.get_w(torch.cat((key, value), dim=-1)).squeeze()  # B * l
        weight = weight.softmax(dim=-1).unsqueeze(dim=-1)  # B * l * 1
        out = (value * weight).sum(dim=1)
        out = self.fc(out)
        return out


class MaskAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.get_w = nn.Sequential(nn.Linear(in_dim * 2, in_dim), nn.ReLU(), nn.Linear(in_dim, 1))

        self.fc = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),)

    def forward(self, value, key, seq_len, maxlen=9):
        # seq_len: (batch,)
        device = value.device
        key = key.unsqueeze(dim=1)
        length = value.shape[1]
        key = key.repeat([1, length, 1])  # (batch, 9, 64)
        weight = self.get_w(torch.cat((key, value), dim=-1)).squeeze(-1)  # (batch, 9)
        mask = sequence_mask(seq_len + 1, maxlen=maxlen, device=device)
        weight[~mask] = float("-inf")
        weight = weight.softmax(dim=-1).unsqueeze(dim=-1)
        out = (value * weight).sum(dim=1)
        out = self.fc(out)
        return out


class TFMaskAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.get_w = nn.Sequential(nn.Linear(in_dim * 2, in_dim), nn.ReLU(), nn.Linear(in_dim, 1))

        self.fc = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),)

    def forward(self, value, key, seq_len, maxlen=9):
        device = value.device
        key = key.unsqueeze(dim=1)
        length = value.shape[1]
        key = key.repeat([1, length, 1])
        weight = self.get_w(torch.cat((key, value), dim=-1)).squeeze(-1)
        mask = sequence_mask(seq_len + 1, maxlen=maxlen, device=device)
        mask = mask.repeat(1, 3)  # (batch, 9*3)
        weight[~mask] = float("-inf")
        weight = weight.softmax(dim=-1).unsqueeze(dim=-1)
        out = (value * weight).sum(dim=1)
        out = self.fc(out)
        return out


class NNAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.q_net = nn.Linear(in_dim, out_dim)
        self.k_net = nn.Linear(in_dim, out_dim)
        self.v_net = nn.Linear(in_dim, out_dim)

    def forward(self, Q, K, V):
        q = self.q_net(Q)
        k = self.k_net(K)
        v = self.v_net(V)

        attn = torch.einsum("ijk,ilk->ijl", q, k)
        attn = attn.to(Q.device)
        attn_prob = torch.softmax(attn, dim=-1)

        attn_vec = torch.einsum("ijk,ikl->ijl", attn_prob, v)

        return attn_vec


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class DARNN(nn.Module):
    def __init__(self, device="cpu", **kargs):
        super().__init__()
        self.emb_dim = kargs["emb_dim"]
        self.hidden_size = kargs["hidden_size"]
        self.num_layers = kargs["num_layers"]
        self.is_bidir = kargs["is_bidir"]
        self.dropout = kargs["dropout"]
        self.seq_len = kargs["seq_len"]
        self.interval = kargs["interval"]
        self.today_length = 238
        self.prev_length = 240
        self.input_length = 480
        self.input_size = 6

        self.rnn = nn.LSTM(
            input_size=self.input_size + self.emb_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.is_bidir,
            dropout=self.dropout,
        )
        self.prev_rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.is_bidir,
            dropout=self.dropout,
        )
        self.fc_out = nn.Linear(in_features=self.hidden_size * 2, out_features=1)
        self.attention = NNAttention(self.hidden_size, self.hidden_size)
        self.act_out = nn.Sigmoid()
        if self.emb_dim != 0:
            self.pos_emb = nn.Embedding(self.input_length, self.emb_dim)

    def forward(self, inputs):
        inputs = inputs.view(-1, self.input_length, self.input_size)  # [B, T, F]
        today_input = inputs[:, : self.today_length, :]
        today_input = torch.cat((torch.zeros_like(today_input[:, :1, :]), today_input), dim=1)
        prev_input = inputs[:, 240 : 240 + self.prev_length, :]
        if self.emb_dim != 0:
            embedding = self.pos_emb(torch.arange(end=self.today_length + 1, device=inputs.device))
            embedding = embedding.repeat([today_input.size()[0], 1, 1])
            today_input = torch.cat((today_input, embedding), dim=-1)
        prev_outs, _ = self.prev_rnn(prev_input)
        today_outs, _ = self.rnn(today_input)

        outs = self.attention(today_outs, prev_outs, prev_outs)
        outs = torch.cat((today_outs, outs), dim=-1)
        outs = outs[:, range(0, self.seq_len * self.interval, self.interval), :]
        # outs = self.fc_out(outs).squeeze()
        return self.act_out(self.fc_out(outs).squeeze(-1)), outs


class Transpose(nn.Module):
    def __init__(self, dim1=0, dim2=1):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class SelfAttention(nn.Module):
    def __init__(self, *args, **kargs):
        super().__init__()
        self.attention = nn.MultiheadAttention(*args, **kargs)

    def forward(self, x):
        return self.attention(x, x, x)[0]


def onehot_enc(y, len):
    y = y.unsqueeze(-1)
    y_onehot = torch.zeros(y.shape[0], len)
    # y_onehot.zero_()
    y_onehot.scatter(1, y, 1)
    return y_onehot


def sequence_mask(lengths, maxlen=None, dtype=torch.bool, device=None):
    if maxlen is None:
        maxlen = lengths.max()
    mask = ~(torch.ones((len(lengths), maxlen), device=device).cumsum(dim=1).t() > lengths).t()
    mask.type(dtype)
    return mask

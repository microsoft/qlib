import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
import sys

from tianshou.data import to_torch


class Teacher_Extractor(nn.Module):
    def __init__(self, device="cpu", feature_size=180, **kargs):
        super().__init__()
        self.device = device
        hidden_size = kargs["hidden_size"]
        fc_size = kargs["fc_size"]
        self.cnn_shape = kargs["cnn_shape"]

        self.rnn = nn.GRU(64, hidden_size, batch_first=True)
        self.rnn2 = nn.GRU(64, hidden_size, batch_first=True)
        self.dnn = nn.Sequential(nn.Linear(2, 64), nn.ReLU(),)
        self.cnn = nn.Sequential(nn.Conv1d(self.cnn_shape[1], 3, 3), nn.ReLU(),)
        self.raw_fc = nn.Sequential(nn.Linear((self.cnn_shape[0] - 2) * 3, 64), nn.ReLU(),)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 32), nn.ReLU(),
        )

    def forward(self, inp):
        inp = to_torch(inp, dtype=torch.float32, device=self.device)
        seq_len = inp[:, -1].to(torch.long)
        batch_size = inp.shape[0]
        raw_in = inp[:, : 6 * 240].reshape(-1, 30, 6).transpose(1, 2) ## public part of state
        dnn_in = inp[:, 6 * 240 : -1].reshape(batch_size, -1, 2) ## private part of state
        cnn_out = self.cnn(raw_in).view(batch_size, 8, -1)
        rnn_in = self.raw_fc(cnn_out)
        rnn2_in = self.dnn(dnn_in)
        rnn2_out = self.rnn2(rnn2_in)[0]
        rnn_out = self.rnn(rnn_in)[0][:, -1, :]
        rnn2_out = rnn2_out[torch.arange(rnn2_out.size(0)), seq_len]
        # dnn_out = self.dnn(dnn_in)
        fc_in = torch.cat((rnn_out, rnn2_out), dim=-1)
        self.feature = self.fc(fc_in)
        return self.feature


class Teacher_Actor(nn.Module):
    def __init__(self, extractor, out_shape, device=torch.device("cpu"), **kargs):
        super().__init__()
        self.extractor = extractor
        self.layer_out = nn.Sequential(nn.Linear(32, out_shape), nn.Softmax(dim=-1))
        self.device = device

    def forward(self, obs, state=None, info={}):
        self.feature = self.extractor(obs)
        out = self.layer_out(self.feature)
        return out, state


class Teacher_Critic(nn.Module):
    def __init__(self, extractor, out_shape, device=torch.device("cpu"), **kargs):
        super().__init__()
        self.extractor = extractor
        self.value_out = nn.Linear(32, 1)
        self.device = device

    def forward(self, obs, state=None, info={}):
        self.feature = self.extractor(obs)
        return self.value_out(self.feature).squeeze(-1)

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
import sys

from tianshou.data import to_torch


class PPO_Extractor(nn.Module):
    def __init__(self, device="cpu", **kargs):
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
        # inp = torch.from_numpy(inp).to(torch.device('cpu'))
        seq_len = inp[:, -1].to(torch.long)
        batch_size = inp.shape[0]
        raw_in = inp[:, : 6 * 240]
        raw_in = torch.cat((torch.zeros_like(inp[:, : 6 * 30]), raw_in), dim=-1)
        raw_in = raw_in.reshape(-1, 30, 6).transpose(1, 2)
        dnn_in = inp[:, -19:-1].reshape(batch_size, -1, 2)
        cnn_out = self.cnn(raw_in).view(batch_size, 9, -1)
        assert not torch.isnan(cnn_out).any()
        rnn_in = self.raw_fc(cnn_out)
        assert not torch.isnan(rnn_in).any()
        rnn2_in = self.dnn(dnn_in)
        assert not torch.isnan(rnn2_in).any()
        rnn2_out = self.rnn2(rnn2_in)[0]
        assert not torch.isnan(rnn2_out).any()
        rnn_out = self.rnn(rnn_in)[0]
        assert not torch.isnan(rnn_out).any()
        rnn_out = rnn_out[torch.arange(rnn_out.size(0)), seq_len]
        rnn2_out = rnn2_out[torch.arange(rnn2_out.size(0)), seq_len]
        # dnn_out = self.dnn(dnn_in)
        fc_in = torch.cat((rnn_out, rnn2_out), dim=-1)
        self.feature = self.fc(fc_in)
        return self.feature


class PPO_Actor(nn.Module):
    def __init__(self, extractor, out_shape, device=torch.device("cpu"), **kargs):
        super().__init__()
        self.extractor = extractor
        self.layer_out = nn.Sequential(nn.Linear(32, out_shape), nn.Softmax(dim=-1))
        self.device = device

    def forward(self, obs, state=None, info={}):
        self.feature = self.extractor(obs)
        assert not (torch.isnan(self.feature).any() | torch.isinf(self.feature).any()), f"{self.feature}"
        out = self.layer_out(self.feature)
        return out, state


class PPO_Critic(nn.Module):
    def __init__(self, extractor, out_shape, device=torch.device("cpu"), **kargs):
        super().__init__()
        self.extractor = extractor
        self.value_out = nn.Linear(32, 1)
        self.device = device

    def forward(self, obs, state=None, info={}):
        self.feature = self.extractor(obs)
        return self.value_out(self.feature).squeeze(dim=-1)

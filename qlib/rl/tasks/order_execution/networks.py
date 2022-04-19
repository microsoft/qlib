# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Dict, Optional, Literal

import torch
import torch.nn as nn
from gym.spaces import Space

from qlib.rl.config import NETWORKS


class Recurrent(nn.Module):
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int = 64,
        output_dim: int = 32,
        rnn_type: Literal['rnn', 'lstm', 'gru'] = 'gru',
        rnn_num_layers: int = 1,
        obs_space: Optional[Space] = None,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_sources = 3

        rnn_classes = {
            'rnn': nn.RNN,
            'lstm': nn.LSTM,
            'gru': nn.GRU
        }

        self.rnn_class = rnn_classes[rnn_type]
        self.rnn_layers = rnn_num_layers

        self.raw_rnn = self.rnn_class(hidden_dim, hidden_dim, batch_first=True, num_layers=self.rnn_layers)
        self.prev_rnn = self.rnn_class(hidden_dim, hidden_dim, batch_first=True, num_layers=self.rnn_layers)
        self.pri_rnn = self.rnn_class(hidden_dim, hidden_dim, batch_first=True, num_layers=self.rnn_layers)

        self.raw_fc = nn.Sequential(nn.Linear(input_dims["data_processed"], hidden_dim), nn.ReLU())
        self.pri_fc = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU())
        self.dire_fc = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        if "feature" in input_dims:
            self.pub_rnn = self.rnn_class(hidden_dim, hidden_dim, batch_first=True, num_layers=self.rnn_layers)
            self.pub_fc = nn.Sequential(nn.Linear(input_dims["feature"], hidden_dim), nn.ReLU())
            self.num_sources += 1

        self._init_extra_branches()

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * self.num_sources, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def _init_extra_branches(self):
        pass

    def _source_features(self, obs, device):
        bs, _, data_dim = obs["data_processed"].size()
        data = torch.cat((torch.zeros(bs, 1, data_dim, device=device), obs["data_processed"]), 1)
        cur_step = obs["cur_step"].long()
        cur_time = obs["cur_time"].long()
        bs_indices = torch.arange(bs, device=device)

        position = obs["position_history"] / obs["target"].unsqueeze(-1)  # [bs, num_step]
        steps = (
            torch.arange(position.size(-1), device=device).unsqueeze(0).repeat(bs, 1).float()
            / obs["num_step"].unsqueeze(-1).float()
        )  # [bs, num_step]
        priv = torch.stack((position.float(), steps), -1)

        data_in = self.raw_fc(data)
        data_out, _ = self.raw_rnn(data_in)
        # as it is padded with zero in front, this should be last minute
        data_out_slice = data_out[bs_indices, cur_time]

        priv_in = self.pri_fc(priv)
        priv_out = self.pri_rnn(priv_in)[0]
        priv_out = priv_out[bs_indices, cur_step]

        sources = [data_out_slice, priv_out]

        if "feature" in obs:
            feature = obs["feature"]  # [bs, num_step, len(feature)]
            feature_in = self.pub_fc(feature)
            feature_out, _ = self.pub_rnn(feature_in)
            feature_out_slice = feature_out[bs_indices, cur_step]
            sources.append(feature_out_slice)
        else:
            feature_out = None

        dir_out = self.dire_fc(torch.stack((obs["acquiring"], 1 - obs["acquiring"]), -1).float())
        sources.append(dir_out)

        return sources, data_out, feature_out

    def forward(self, inp):
        """
        Input should be a dict containing:

        - data_processed: [N, T, C]
        - cur_step: [N]  (int)
        - cur_time: [N]  (int)
        - position_history: [N, S]  (S is number of steps)
        - target: [N]
        - num_step: [N]  (int)
        - feature: [N, S, C]
        - acquiring: [N]  (0 or 1)
        """

        device = inp["data_processed"].device

        sources, _, __ = self._source_features(inp, device)
        assert len(sources) == self.num_sources

        out = torch.cat(sources, -1)
        return self.fc(out)

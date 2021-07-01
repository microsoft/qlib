# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class MLPModel(nn.Module):
    def __init__(self, d_feat, hidden_size=256, num_layers=3, dropout=0.0, output_dim=1):
        super().__init__()

        self.mlp = nn.Sequential()
        self.softmax = nn.Softmax(dim=1)

        for i in range(num_layers):
            if i > 0:
                self.mlp.add_module("drop_%d" % i, nn.Dropout(dropout))
            self.mlp.add_module("fc_%d" % i, nn.Linear(d_feat if i == 0 else hidden_size, hidden_size))
            self.mlp.add_module("relu_%d" % i, nn.ReLU())

        self.mlp.add_module("fc_out", nn.Linear(hidden_size, output_dim))

    def forward(self, x):
        # feature
        # [N, F]
        out = self.mlp(x).squeeze()
        out = self.softmax(out)
        return out


class GRUModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat

    def forward(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()

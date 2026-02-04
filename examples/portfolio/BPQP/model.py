import torch
from torch import nn

torch.set_default_dtype(torch.float32)
device = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    def __init__(self, d_feat, hidden_size=128, num_layers=3, dropout=0.0):
        super().__init__()

        self.mlp = nn.Sequential()
        for i in range(num_layers):
            if i > 0:
                self.mlp.add_module("drop_%d" % i, nn.Dropout(dropout))
            self.mlp.add_module("fc_%d" % i, nn.Linear(d_feat if i == 0 else hidden_size, hidden_size))
            self.mlp.add_module("bd_%d" % i, nn.BatchNorm1d(hidden_size))
            self.mlp.add_module("relu_%d" % i, nn.ReLU())

        self.mlp.add_module("fc_out", nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.mlp(x).squeeze()


class ECONet:
    pass

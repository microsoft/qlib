import collections
import math

import torch
from qlib.model import Model

from qlib.utils import init_instance_by_config

from torch import nn
from torch.nn import functional as F, init


def cosine(x1, x2, eps=1e-8):
    x1 = x1 / (torch.norm(x1, p=2, dim=-1, keepdim=True) + eps)
    x2 = x2 / (torch.norm(x2, p=2, dim=-1, keepdim=True) + eps)
    return x1 @ x2.transpose(0, 1)


# class LabelAdaptHead(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.weight = nn.Parameter(torch.empty(1))
#         self.bias = nn.Parameter(torch.ones(1) / 8)
#         init.uniform_(self.weight, 0.75, 1.25)
#
#     def forward(self, y, inverse=False):
#         if inverse:
#             return (y - self.bias) / (self.weight + 1e-9)
#         else:
#             return (self.weight + 1e-9) * y + self.bias

class LabelAdaptHeads(nn.Module):
    def __init__(self, num_head):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, num_head))
        self.bias = nn.Parameter(torch.ones(1, num_head) / 8)
        init.uniform_(self.weight, 0.75, 1.25)

    def forward(self, y, inverse=False):
        if inverse:
            return (y.view(-1, 1) - self.bias) / (self.weight + 1e-9)
        else:
            return (self.weight + 1e-9) * y.view(-1, 1) + self.bias

class LabelAdapter(nn.Module):
    def __init__(self, x_dim, num_head=4, temperature=4, hid_dim=32):
        super().__init__()
        self.num_head = num_head
        self.linear = nn.Linear(x_dim, hid_dim, bias=False)
        self.P = nn.Parameter(torch.empty(num_head, hid_dim))
        init.kaiming_uniform_(self.P, a=math.sqrt(5))
        # self.heads = nn.ModuleList([LabelAdaptHead() for _ in range(num_head)])
        self.heads = LabelAdaptHeads(num_head)
        self.temperature = temperature

    def forward(self, x, y, inverse=False):
        v = self.linear(x.reshape(len(x), -1))
        gate = cosine(v, self.P)
        gate = torch.softmax(gate / self.temperature, -1)
        # return sum([gate[:, i] * self.heads[i](y, inverse=inverse) for i in range(self.num_head)])
        return (gate * self.heads(y, inverse=inverse)).sum(-1)


class FeatureAdapter(nn.Module):
    def __init__(self, in_dim, num_head=4, temperature=4):
        super().__init__()
        self.num_head = num_head
        self.P = nn.Parameter(torch.empty(num_head, in_dim))
        init.kaiming_uniform_(self.P, a=math.sqrt(5))
        self.heads = nn.ModuleList([nn.Linear(in_dim, in_dim, bias=True) for _ in range(num_head)])
        self.temperature = temperature

    def forward(self, x):
        s_hat = torch.cat(
            [torch.cosine_similarity(x, self.P[i], dim=-1).unsqueeze(-1) for i in range(self.num_head)], -1,
        )
        # s_hat = cosine(x, self.P)
        s = torch.softmax(s_hat / self.temperature, -1).unsqueeze(-1)
        return x + sum([s[..., i, :] * self.heads[i](x) for i in range(self.num_head)])


class ForecastModel(nn.Module):
    def __init__(self, task_config, x_dim=None, lr=0.001, need_permute=False, model=None):
        super().__init__()
        self.lr = lr
        # self.lr = task_config["model"]['kwargs']['lr']
        self.criterion = nn.MSELoss()
        if task_config["model"]["class"] == "LinearModel":
            self.model = nn.Linear(x_dim, 1)
            self.model.load_state_dict(
                collections.OrderedDict(
                    {"weight": torch.from_numpy(model.coef_).unsqueeze(0), "bias": torch.tensor([model.intercept_]),}
                )
            )
            self.opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
            self.device = None
        else:
            if model is None:
                model = init_instance_by_config(task_config["model"], accept_types=Model)
                self.opt = None
            else:
                self.opt = model.train_optimizer
            for child in model.__dict__.values():
                if isinstance(child, nn.Module):
                    self.model = child
                    break
            if hasattr(model, "device"):
                self.device = model.device
            else:
                self.device = torch.device("cuda")
            self.need_permute = need_permute
        if self.opt is None:
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.device is not None:
            self.to(self.device)

    def forward(self, X, model=None):
        if model is None:
            model = self.model
        if X.dim() == 3:
            X = X.permute(0, 2, 1).reshape(len(X), -1) if self.need_permute else X.reshape(len(X), -1)
        y_hat = model(X)
        y_hat = y_hat.view(-1)
        return y_hat


class DoubleAdapt(ForecastModel):
    def __init__(
        self, task_config, factor_num, x_dim=None, lr=0.001, need_permute=False, model=None, num_head=8, temperature=10,
    ):
        super().__init__(
            task_config=task_config, x_dim=x_dim, lr=lr, need_permute=need_permute, model=model,
        )
        self.teacher_x = FeatureAdapter(factor_num, num_head, temperature)
        self.teacher_y = LabelAdapter(factor_num if x_dim is None else x_dim, num_head, temperature)
        self.meta_params = list(self.teacher_x.parameters()) + list(self.teacher_y.parameters())
        if self.device is not None:
            self.to(self.device)

    def forward(self, X, model=None, transform=False):
        if transform:
            X = self.teacher_x(X)
        return super().forward(X, model), X



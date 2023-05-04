# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
from torch import nn

from qlib.constant import EPS
from qlib.log import get_module_logger


class ICLoss(nn.Module):
    def forward(self, pred, y, idx, skip_size=50):
        """forward.
        FIXME:
        - Some times it will be a slightly different from the result from `pandas.corr()`
        - It may be caused by the precision problem of model;

        :param pred:
        :param y:
        :param idx: Assume the level of the idx is (date, inst), and it is sorted
        """
        prev = None
        diff_point = []
        for i, (date, inst) in enumerate(idx):
            if date != prev:
                diff_point.append(i)
            prev = date
        diff_point.append(None)
        # The lengths of diff_point will be one more larger then diff_point

        ic_all = 0.0
        skip_n = 0
        for start_i, end_i in zip(diff_point, diff_point[1:]):
            pred_focus = pred[start_i:end_i]  # TODO: just for fake
            if pred_focus.shape[0] < skip_size:
                # skip some days which have very small amount of stock.
                skip_n += 1
                continue
            y_focus = y[start_i:end_i]
            if pred_focus.std() < EPS or y_focus.std() < EPS:
                # These cases often happend at the end of test data.
                # Usually caused by fillna(0.)
                skip_n += 1
                continue

            ic_day = torch.dot(
                (pred_focus - pred_focus.mean()) / np.sqrt(pred_focus.shape[0]) / pred_focus.std(),
                (y_focus - y_focus.mean()) / np.sqrt(y_focus.shape[0]) / y_focus.std(),
            )
            ic_all += ic_day
        if len(diff_point) - 1 - skip_n <= 0:
            raise ValueError("No enough data for calculating IC")
        if skip_n > 0:
            get_module_logger("ICLoss").info(
                f"{skip_n} days are skipped due to zero std or small scale of valid samples."
            )
        ic_mean = ic_all / (len(diff_point) - 1 - skip_n)
        return -ic_mean  # ic loss


def preds_to_weight_with_clamp(preds, clip_weight=None, clip_method="tanh"):
    """
    Clip the weights.

    Parameters
    ----------
    clip_weight: float
        The clip threshold.
    clip_method: str
        The clip method. Current available: "clamp", "tanh", and "sigmoid".
    """
    if clip_weight is not None:
        if clip_method == "clamp":
            weights = torch.exp(preds)
            weights = weights.clamp(1.0 / clip_weight, clip_weight)
        elif clip_method == "tanh":
            weights = torch.exp(torch.tanh(preds) * np.log(clip_weight))
        elif clip_method == "sigmoid":
            # intuitively assume its sum is 1
            if clip_weight == 0.0:
                weights = torch.ones_like(preds)
            else:
                sm = nn.Sigmoid()
                weights = sm(preds) * clip_weight  # TODO: The clip_weight is useless here.
                weights = weights / torch.sum(weights) * weights.numel()
        else:
            raise ValueError("Unknown clip_method")
    else:
        weights = torch.exp(preds)
    return weights


class SingleMetaBase(nn.Module):
    def __init__(self, hist_n, clip_weight=None, clip_method="clamp"):
        # method can be tanh or clamp
        super().__init__()
        self.clip_weight = clip_weight
        if clip_method in ["tanh", "clamp"]:
            if self.clip_weight is not None and self.clip_weight < 1.0:
                self.clip_weight = 1 / self.clip_weight
        self.clip_method = clip_method

    def is_enabled(self):
        if self.clip_weight is None:
            return True
        if self.clip_method == "sigmoid":
            if self.clip_weight > 0.0:
                return True
        else:
            if self.clip_weight > 1.0:
                return True
        return False

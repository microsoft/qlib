# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from hyperopt import hp


TopkAmountStrategySpace = {
    "topk": hp.choice("topk", [30, 35, 40]),
    "buffer_margin": hp.choice("buffer_margin", [200, 250, 300]),
}

QLibDataLabelSpace = {
    "labels": hp.choice(
        "labels",
        [["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["Ref($close, -5)/$close - 1"]],
    )
}

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
import copy
import logging

from .net import MLPModel

from ....data.dataset import DatasetH


class MetaModelTCTS(MetaGuideModel):
    """
    The meta-model for TCTS
    """

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        batch_size=2000,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        output_dim=5,
        lr=5e-7,
        steps=3,
        GPU=0,
        seed=None,
        target_label=0,
        **kwargs
    ):
        # Set logger.
        self.logger = get_module_logger("TCTS")
        self.logger.info("TCTS pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.loss = loss
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() else "cpu")
        self.use_gpu = torch.cuda.is_available()
        self.seed = seed
        self.output_dim = output_dim
        self.lr = lr
        self.steps = steps
        self.target_label = target_label

        self.logger.info(
            "TCTS parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\nloss_type : {}"
            "\nvisible_GPU : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                batch_size,
                early_stop,
                loss,
                GPU,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.weight_model = MLPModel(
            d_feat=360 + 2 * self.output_dim + 1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_dim=self.output_dim,
        )
        if optimizer.lower() == "adam":
            self.optimizer = optim.Adam(self.weight_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.optimizer = optim.SGD(self.weight_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.weight_model.to(self.device)

    def loss_fn(self, pred, label, weight):

        loc = torch.argmax(weight, 1)
        loss = (pred - label[np.arange(weight.shape[0]), loc]) ** 2
        return torch.mean(loss)

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        verbose=True,
        save_path=None,
    ):
        pass

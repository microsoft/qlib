#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import re
import pandas as pd
from sklearn.metrics import mean_squared_error
from pprint import pprint
import numpy as np

from ...workflow.record_temp import SignalRecord
from ...log import get_module_logger

logger = get_module_logger("workflow", "INFO")


class SignalMseRecord(SignalRecord):
    """
    This is the Signal MSE Record class that computes the mean squared error (MSE).
    This class inherits the ``SignalMseRecord`` class.
    """

    artifact_path = "sig_analysis"

    def __init__(self, recorder, **kwargs):
        super().__init__(recorder=recorder, **kwargs)

    def generate(self, **kwargs):
        try:
            self.check(parent=True)
        except FileExistsError:
            super().generate()

        pred = self.load("pred.pkl")
        label = self.load("label.pkl")
        masks = ~np.isnan(label.values)
        mse = mean_squared_error(pred.values[masks], label[masks])
        metrics = {"MSE": mse, "RMSE": np.sqrt(mse)}
        objects = {"mse.pkl": mse, "rmse.pkl": np.sqrt(mse)}
        self.recorder.log_metrics(**metrics)
        self.recorder.save_objects(**objects, artifact_path=self.get_path())
        pprint(metrics)

    def list(self):
        paths = [self.get_path("mse.pkl"), self.get_path("rmse.pkl")]
        return paths

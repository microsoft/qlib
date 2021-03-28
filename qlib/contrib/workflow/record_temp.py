#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import re
import pandas as pd
from sklearn.metrics import mean_squared_error
from pprint import pprint
from typing import Dict, Text, Any
import numpy as np

from ...workflow.record_temp import RecordTemp
from ...workflow.record_temp import SignalRecord
from ...data import dataset as qlib_dataset
from ...log import get_module_logger

logger = get_module_logger("workflow", "INFO")


class MultiSegRecord(RecordTemp):
    """
    This is the multiple segments signal record class that generates the signal prediction.
    This class inherits the ``RecordTemp`` class.
    """

    def __init__(self, model, dataset, recorder=None):
        super().__init__(recorder=recorder)
        if not isinstance(dataset, qlib_dataset.DatasetH):
            raise ValueError("The type of dataset is not DatasetH instead of {:}".format(type(dataset)))
        self.model = model
        self.dataset = dataset

    def generate(self, segments: Dict[Text, Any], save: bool = False):
        # generate prediciton
        for key, segment in segments.items():
            predics = self.model.predict(self.dataset, segment)
            if isinstance(pred, pd.Series):
                predics = predictions.to_frame("score")
            # self.recorder.save_objects(**{"pred.pkl": pred})
            labels = self.dataset.prepare(
                segments=segment, col_set="label", data_key=dataset.handler.DataHandlerLP.DK_R
            )
            # compute ic, rank_ic


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

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This example shows how OnlineTool works when we need update prediction.
There are two parts including first_train and update_online_pred.
Firstly, we will finish the training and set the trained models to the `online` models.
Next, we will finish updating online predictions.
"""
import copy
import fire
import qlib
from qlib.config import REG_CN
from qlib.model.trainer import task_train
from qlib.workflow.online.utils import OnlineToolR
from qlib.tests.config import CSI300_GBDT_TASK

task = copy.deepcopy(CSI300_GBDT_TASK)

task["record"] = {
    "class": "SignalRecord",
    "module_path": "qlib.workflow.record_temp",
}


class UpdatePredExample:
    def __init__(
        self, provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN, experiment_name="online_srv", task_config=task
    ):
        qlib.init(provider_uri=provider_uri, region=region)
        self.experiment_name = experiment_name
        self.online_tool = OnlineToolR(self.experiment_name)
        self.task_config = task_config

    def first_train(self):
        rec = task_train(self.task_config, experiment_name=self.experiment_name)
        self.online_tool.reset_online_tag(rec)  # set to online model

    def update_online_pred(self):
        self.online_tool.update_online_pred()

    def main(self):
        self.first_train()
        self.update_online_pred()


if __name__ == "__main__":
    ## to train a model and set it to online model, use the command below
    # python update_online_pred.py first_train
    ## to update online predictions once a day, use the command below
    # python update_online_pred.py update_online_pred
    ## to see the whole process with your own parameters, use the command below
    # python update_online_pred.py main --experiment_name="your_exp_name"
    fire.Fire(UpdatePredExample)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
The motivation of this demo
- To show the data modules of Qlib is Serializable, users can dump processed data to disk to avoid duplicated data preprocessing
"""

from copy import deepcopy
from pathlib import Path
import pickle
from pprint import pprint
from ruamel.yaml import YAML
import subprocess

from qlib import init
from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import TimeInspector
from qlib.model.trainer import task_train
from qlib.utils import init_instance_by_config

# For general purpose, we use relative path
DIRNAME = Path(__file__).absolute().resolve().parent

if __name__ == "__main__":
    init()

    repeat = 2
    exp_name = "data_mem_reuse_demo"

    config_path = DIRNAME.parent / "benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml"
    yaml = YAML(typ="safe", pure=True)
    task_config = yaml.load(config_path.open())

    # 1) without using processed data in memory
    with TimeInspector.logt("The original time without reusing processed data in memory:"):
        for i in range(repeat):
            task_train(task_config["task"], experiment_name=exp_name)

    # 2) prepare processed data in memory.
    hd_conf = task_config["task"]["dataset"]["kwargs"]["handler"]
    pprint(hd_conf)
    hd: DataHandlerLP = init_instance_by_config(hd_conf)

    # 3) with reusing processed data in memory
    new_task = deepcopy(task_config["task"])
    new_task["dataset"]["kwargs"]["handler"] = hd
    print(new_task)

    with TimeInspector.logt("The time with reusing processed data in memory:"):
        # this will save the time to reload and process data from disk(in `DataHandlerLP`)
        # It still takes a lot of time in the backtest phase
        for i in range(repeat):
            task_train(new_task, experiment_name=exp_name)

    # 4) User can change other parts exclude processed data in memory(handler)
    new_task = deepcopy(task_config["task"])
    new_task["dataset"]["kwargs"]["segments"]["train"] = ("20100101", "20131231")
    with TimeInspector.logt("The time with reusing processed data in memory:"):
        task_train(new_task, experiment_name=exp_name)

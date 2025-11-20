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
from qlib.log import TimeInspector

from qlib import init
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config

# For general purpose, we use relative path
DIRNAME = Path(__file__).absolute().resolve().parent

if __name__ == "__main__":
    init()

    config_path = DIRNAME.parent / "benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml"

    # 1) show original time
    with TimeInspector.logt("The original time without handler cache:"):
        subprocess.run(f"qrun {config_path}", shell=True)

    # 2) dump handler
    yaml = YAML(typ="safe", pure=True)
    task_config = yaml.load(config_path.open())
    hd_conf = task_config["task"]["dataset"]["kwargs"]["handler"]
    pprint(hd_conf)
    hd: DataHandlerLP = init_instance_by_config(hd_conf)
    hd_path = DIRNAME / "handler.pkl"
    hd.to_pickle(hd_path, dump_all=True)

    # 3) create new task with handler cache
    new_task_config = deepcopy(task_config)
    new_task_config["task"]["dataset"]["kwargs"]["handler"] = f"file://{hd_path}"
    new_task_config["sys"] = {"path": [str(config_path.parent.resolve())]}
    new_task_path = DIRNAME / "new_task.yaml"
    print("The location of the new task", new_task_path)

    # save new task
    with new_task_path.open("w") as f:
        yaml.safe_dump(new_task_config, f, indent=4, sort_keys=False)

    # 4) train model with new task
    with TimeInspector.logt("The time for task with handler cache:"):
        subprocess.run(f"qrun {new_task_path}", shell=True)

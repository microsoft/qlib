#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import sys, os
from pathlib import Path

import qlib
import fire
import pandas as pd
import ruamel.yaml as yaml
from qlib.config import C
from qlib.model.trainer import task_train


def get_path_list(path):
    if isinstance(path, str):
        return [path]
    else:
        return list(path)


def sys_config(config, config_path):
    """
    Configure the `sys` section

    Parameters
    ----------
    config : dict
        configuration of the workflow.
    config_path : str
        path of the configuration
    """
    sys_config = config.get("sys", {})

    # abspath
    for p in get_path_list(sys_config.get("path", [])):
        sys.path.append(p)

    # relative path to config path
    for p in get_path_list(sys_config.get("rel_path", [])):
        sys.path.append(str(Path(config_path).parent.resolve().absolute() / p))


# workflow handler function
def workflow(config_path, experiment_name="workflow", uri_folder="mlruns"):
    with open(config_path) as fp:
        config = yaml.safe_load(fp)

    # config the `sys` section
    sys_config(config, config_path)

    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str(Path(os.getcwd()).resolve() / uri_folder)
    qlib.init(**config.get("qlib_init"), exp_manager=exp_manager)

    recorder = task_train(config.get("task"), experiment_name=experiment_name)
    recorder.save_objects(config=config)


# function to run workflow by config
def run():
    fire.Fire(workflow)


if __name__ == "__main__":
    run()

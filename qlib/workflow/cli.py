#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import sys
from pathlib import Path

import qlib
import fire
import pandas as pd
import ruamel.yaml as yaml
from qlib.model.trainer import task_train


def get_path_list(path):
    if isinstance(path, str):
        return [path]
    else:
        return [p for p in path]


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


# worflow handler function
def workflow(config_path, experiment_name="workflow"):
    with open(config_path) as fp:
        config = yaml.load(fp, Loader=yaml.Loader)

    # config the `sys` section
    sys_config(config, config_path)

    provider_uri = config.get("provider_uri")
    region = config.get("region")
    qlib.init(provider_uri=provider_uri, region=region)

    task_train(config, experiment_name=experiment_name)


# function to run worklflow by config
def run():
    fire.Fire(workflow)


if __name__ == "__main__":
    run()

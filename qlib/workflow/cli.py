#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import sys
from pathlib import Path

import qlib
import fire
import pandas as pd
import ruamel.yaml as yaml
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord


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
        configuration of the path.
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

    # model initiaiton
    model = init_instance_by_config(config.get("task")["model"])
    dataset = init_instance_by_config(config.get("task")["dataset"])

    # start exp
    with R.start(experiment_name=experiment_name):
        # train model
        R.log_params(**flatten_dict(config.get("task")))
        model.fit(dataset)
        recorder = R.get_recorder()

        # generate records: prediction, backtest, and analysis
        for record in config.get("task")["record"]:
            if record["class"] == SignalRecord.__name__:
                srconf = {"model": model, "dataset": dataset, "recorder": recorder}
                record["kwargs"].update(srconf)
                sr = init_instance_by_config(record)
                sr.generate()
            else:
                rconf = {"recorder": recorder}
                record["kwargs"].update(rconf)
                ar = init_instance_by_config(record)
                ar.generate()


# function to run worklflow by config
def run():
    fire.Fire(workflow)


if __name__ == "__main__":
    run()

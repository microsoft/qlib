#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import sys
from pathlib import Path

import qlib
import fire
import pandas as pd
import ruamel.yaml as yaml
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord

# worflow handler function
def workflow(config_path):
    with open(config_path) as fp:
        config = yaml.load(fp, Loader=yaml.Loader)

    provider_uri = config.get("provider_uri")
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # model initiaiton
    model = init_instance_by_config(config.get("task")["model"])
    dataset = init_instance_by_config(config.get("task")["dataset"])

    # start exp
    with R.start(experiment_name="workflow"):
        R.log_paramters(**flatten_dict(task))
        model.fit(dataset)
        recorder = R.get_recorder()

        # generate records
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

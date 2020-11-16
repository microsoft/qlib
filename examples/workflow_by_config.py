#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import sys
from pathlib import Path

import qlib
import fire
import yaml
import pandas as pd
from qlib.config import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord

# worflow handler function
def workflow(config_path):
    with open(config_path) as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    provider_uri = config.get("PROVIDER_URI")
    if not exists_qlib_data(provider_uri):
        print(f"Qlib data is not found in {provider_uri}")
        sys.path.append(str(Path(__file__).resolve().parent.parent.joinpath("scripts")))
        from get_data import GetData

        GetData().qlib_data_cn(target_dir=provider_uri)

    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # model initiaiton
    model = init_instance_by_config(config.get("TASK")["model"])
    dataset = init_instance_by_config(config.get("TASK")["dataset"])

    # start exp
    with R.start("workflow"):
        model.fit(dataset)

        # prediction
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # backtest
        par = PortAnaRecord(recorder, config.get("PORT_ANALYSIS_CONFIG"))
        par.generate()

if __name__ == "__main__":
    fire.Fire(workflow)
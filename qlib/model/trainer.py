# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord


def task_train(config: dict, experiment_name):
    """
    task based training

    Parameters
    ----------
    config : dict
        A dict describing the training process
    """

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

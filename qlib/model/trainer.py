# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord


def task_train(task_config: dict, experiment_name):
    """
    task based training

    Parameters
    ----------
    task_config : dict
        A dict describes a task setting.
    """

    # model initiaiton
    model = init_instance_by_config(task_config["model"])
    dataset = init_instance_by_config(task_config["dataset"])

    # start exp
    with R.start(experiment_name=experiment_name):
        # train model
        R.log_params(**flatten_dict(task_config))
        model.fit(dataset)
        recorder = R.get_recorder()
        R.save_objects(**{"params.pkl": model})

        # generate records: prediction, backtest, and analysis
        for record in task_config["record"]:
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

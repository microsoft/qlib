# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord


def task_train(task_config: dict, experiment_name: str) -> str:
    """
    task based training

    Parameters
    ----------
    task_config : dict
        A dict describes a task setting.
    experiment_name: str
        The name of experiment

    Returns
    ----------
    rid : str
        The id of the recorder of this task
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
        R.save_objects(**{"task.pkl": task_config})  # keep the original format and datatype

        # generate records: prediction, backtest, and analysis
        records = task_config.get("record", [])
        if isinstance(records, dict):  # prevent only one dict
            records = [records]
        for record in records:
            if record["class"] == SignalRecord.__name__:
                srconf = {"model": model, "dataset": dataset, "recorder": recorder}
                record.setdefault("kwargs", {})
                record["kwargs"].update(srconf)
                sr = init_instance_by_config(record)
                sr.generate()
            else:
                rconf = {"recorder": recorder}
                record.setdefault("kwargs", {})
                record["kwargs"].update(rconf)
                ar = init_instance_by_config(record)
                ar.generate()
    return recorder.info["id"]

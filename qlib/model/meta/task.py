# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from qlib.data.dataset import Dataset
from ...utils import init_instance_by_config


class MetaTask:
    """
    A single meta-task, a meta-dataset contains a list of them.
    It serves as a component as in MetaDatasetDS

    The data processing is different

    - the processed input may be different between training and testing

        - When training, the X, y, X_test, y_test in training tasks are necessary (# PROC_MODE_FULL #)
          but not necessary in test tasks. (# PROC_MODE_TEST #)
        - When the meta model can be transferred into other dataset, only meta_info is necessary  (# PROC_MODE_TRANSFER #)
    """

    PROC_MODE_FULL = "full"
    PROC_MODE_TEST = "test"
    PROC_MODE_TRANSFER = "transfer"

    def __init__(self, task: dict, meta_info: object, mode: str = PROC_MODE_FULL):
        """
        The `__init__` func is responsible for

        - store the task
        - store the origin input data for
        - process the input data for meta data

        Parameters
        ----------
        task : dict
            the task to be enhanced by meta model

        meta_info : object
            the input for meta model
        """
        self.task = task
        self.meta_info = meta_info  # the original meta input information, it will be processed later
        self.mode = mode

    def get_dataset(self) -> Dataset:
        return init_instance_by_config(self.task["dataset"], accept_types=Dataset)

    def get_meta_input(self) -> object:
        """
        Return the **processed** meta_info
        """
        return self.meta_info

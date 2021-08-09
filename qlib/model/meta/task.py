# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from typing import Union, List, Tuple

from qlib.data.dataset import Dataset
from ...utils import init_instance_by_config


class MetaTask:
    """
    A single meta-task, a meta-dataset contains a list of them.
    It is designed for Mea
    """

    def __init__(self, task: dict, meta_info: object):
        """
        the `__init__` func is responsible for
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

    def get_dataset(self) -> Dataset:
        return init_instance_by_config(self.task["dataset"], accept_types=Dataset)

    def get_meta_input(elf) -> object:
        """
        Return the **processed** meta_info
        """
        return self.meta_info

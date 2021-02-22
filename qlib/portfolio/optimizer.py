# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc


class BaseOptimizer(abc.ABC):
    """Modeling things"""

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> object:
        """ Generate a optimized portfolio allocation """
        pass

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc


class BaseOptimizer(abc.ABC):
    """Construct portfolio with a optimization related method"""

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> object:
        """Generate a optimized portfolio allocation"""
        pass

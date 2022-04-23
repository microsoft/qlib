# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Defines a set of initial state definitions and state-set definitions.

With single-asset order execution only, the only seed is order.
"""

from typing import TypeVar

InitialStateType = TypeVar("InitialStateType")
"""Type of data that creates the simulator."""

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Commonly used types."""

import sys

from typing import TYPE_CHECKING

__all__ = ["Literal", "TypedDict", "final"]

if TYPE_CHECKING or sys.version_info >= (3, 8):
    from typing import Literal, TypedDict, final  # type: ignore
else:
    from typing_extensions import Literal, TypedDict, final

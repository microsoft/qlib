# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Commonly used types."""

import sys

from typing import TYPE_CHECKING

__all__ = ["Literal", "TypedDict", "final"]

if sys.version_info >= (3, 8):
    from typing import Literal, TypedDict, final  # type: ignore  # pylint: disable=no-name-in-module
else:
    from typing_extensions import Literal, TypedDict, final

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Commonly used types."""

import sys

__all__ = ["Literal"]

if sys.version_info >= (3, 8):
    from typing import Literal  # type: ignore
else:
    from typing_extensions import Literal

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Commonly used types."""

__all__ = ['Literal']

try:
    from typing import Literal
except:
    from typing_extensions import Literal

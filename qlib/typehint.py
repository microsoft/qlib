# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Commonly used types."""

import sys
from typing import Union
from pathlib import Path

__all__ = ["Literal", "TypedDict", "final"]

if sys.version_info >= (3, 8):
    from typing import Literal, TypedDict, final  # type: ignore  # pylint: disable=no-name-in-module
else:
    from typing_extensions import Literal, TypedDict, final


class InstDictConf(TypedDict):
    """
    InstDictConf  is a Dict-based config to describe an instance

        case 1)
        {
            'class': 'ClassName',
            'kwargs': dict, #  It is optional. {} will be used if not given
            'model_path': path, # It is optional if module is given in the class
        }
        case 2)
        {
            'class': <The class it self>,
            'kwargs': dict, #  It is optional. {} will be used if not given
        }
    """

    # class: str  # because class is a keyword of Python. We have to comment it
    kwargs: dict  # It is optional. {} will be used if not given
    module_path: str  # It is optional if module is given in the class


InstConf = Union[InstDictConf, str, object, Path]
"""
InstConf is a type to describe an instance; it will be passed into init_instance_by_config for Qlib

    config : Union[str, dict, object, Path]

        InstDictConf example.
            please refer to the docs of InstDictConf

        str example.
            1) specify a pickle object
                - path like 'file:///<path to pickle file>/obj.pkl'
            2) specify a class name
                - "ClassName":  getattr(module, "ClassName")() will be used.
            3) specify module path with class name
                - "a.b.c.ClassName" getattr(<a.b.c.module>, "ClassName")() will be used.

        object example:
            instance of accept_types

        Path example:
            specify a pickle object
                - it will be treated like 'file:///<path to pickle file>/obj.pkl'
"""

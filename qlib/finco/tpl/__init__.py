# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path

DIRNAME = Path(__file__).absolute().resolve().parent


def get_tpl_path() -> Path:
    """
    return the template path
    Because the template path is located in the folder. We don't know where it is located. So __file__ for this module will be used.
    """
    return DIRNAME

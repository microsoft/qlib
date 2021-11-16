# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# TODO: move file related utils into this module
import contextlib
from typing import IO, Union
from pathlib import Path


@contextlib.contextmanager
def get_io_object(file: Union[IO, str, Path], *args, **kwargs) -> IO:
    """
    providing a easy interface to get an IO object

    Parameters
    ----------
    file : Union[IO, str, Path]
        a object representing the file

    Returns
    -------
    IO:
        a IO-like object

    Raises
    ------
    NotImplementedError:
    """
    if isinstance(file, IO):
        yield file
    else:
        if isinstance(file, str):
            file = Path(file)
        if not isinstance(file, Path):
            raise NotImplementedError(f"This type[{type(file)}] of input is not supported")
        with file.open(*args, **kwargs) as f:
            yield f

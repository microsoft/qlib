# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import contextlib
import os
import shutil
import tempfile
import contextlib
from typing import Optional, Text, IO, Union
from pathlib import Path

from qlib.log import get_module_logger

log = get_module_logger("utils.file")


def get_or_create_path(path: Optional[Text] = None, return_dir: bool = False):
    """Create or get a file or directory given the path and return_dir.

    Parameters
    ----------
    path: a string indicates the path or None indicates creating a temporary path.
    return_dir: if True, create and return a directory; otherwise c&r a file.

    """
    if path:
        if return_dir and not os.path.exists(path):
            os.makedirs(path)
        elif not return_dir:  # return a file, thus we need to create its parent directory
            xpath = os.path.abspath(os.path.join(path, ".."))
            if not os.path.exists(xpath):
                os.makedirs(xpath)
    else:
        temp_dir = os.path.expanduser("~/tmp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        if return_dir:
            _, path = tempfile.mkdtemp(dir=temp_dir)
        else:
            _, path = tempfile.mkstemp(dir=temp_dir)
    return path


@contextlib.contextmanager
def save_multiple_parts_file(filename, format="gztar"):
    """Save multiple parts file

    Implementation process:
        1. get the absolute path to 'filename'
        2. create a 'filename' directory
        3. user does something with file_path('filename/')
        4. remove 'filename' directory
        5. make_archive 'filename' directory, and rename 'archive file' to filename

    :param filename: result model path
    :param format: archive format: one of "zip", "tar", "gztar", "bztar", or "xztar"
    :return: real model path

    Usage::

        >>> # The following code will create an archive file('~/tmp/test_file') containing 'test_doc_i'(i is 0-10) files.
        >>> with save_multiple_parts_file('~/tmp/test_file') as filename_dir:
        ...   for i in range(10):
        ...       temp_path = os.path.join(filename_dir, 'test_doc_{}'.format(str(i)))
        ...       with open(temp_path) as fp:
        ...           fp.write(str(i))
        ...

    """

    if filename.startswith("~"):
        filename = os.path.expanduser(filename)

    file_path = os.path.abspath(filename)

    # Create model dir
    if os.path.exists(file_path):
        raise FileExistsError("ERROR: file exists: {}, cannot be create the directory.".format(file_path))

    os.makedirs(file_path)

    # return model dir
    yield file_path

    # filename dir to filename.tar.gz file
    tar_file = shutil.make_archive(file_path, format=format, root_dir=file_path)

    # Remove filename dir
    if os.path.exists(file_path):
        shutil.rmtree(file_path)

    # filename.tar.gz rename to filename
    os.rename(tar_file, file_path)


@contextlib.contextmanager
def unpack_archive_with_buffer(buffer, format="gztar"):
    """Unpack archive with archive buffer
    After the call is finished, the archive file and directory will be deleted.

    Implementation process:
        1. create 'tempfile' in '~/tmp/' and directory
        2. 'buffer' write to 'tempfile'
        3. unpack archive file('tempfile')
        4. user does something with file_path('tempfile/')
        5. remove 'tempfile' and 'tempfile directory'

    :param buffer: bytes
    :param format: archive format: one of "zip", "tar", "gztar", "bztar", or "xztar"
    :return: unpack archive directory path

    Usage::

        >>> # The following code is to print all the file names in 'test_unpack.tar.gz'
        >>> with open('test_unpack.tar.gz') as fp:
        ...     buffer = fp.read()
        ...
        >>> with unpack_archive_with_buffer(buffer) as temp_dir:
        ...     for f_n in os.listdir(temp_dir):
        ...         print(f_n)
        ...

    """
    temp_dir = os.path.expanduser("~/tmp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=temp_dir) as fp:
        fp.write(buffer)
        file_path = fp.name

    try:
        tar_file = file_path + ".tar.gz"
        os.rename(file_path, tar_file)
        # Create dir
        os.makedirs(file_path)
        shutil.unpack_archive(tar_file, format=format, extract_dir=file_path)

        # Return temp dir
        yield file_path

    except Exception as e:
        log.error(str(e))
    finally:
        # Remove temp tar file
        if os.path.exists(tar_file):
            os.unlink(tar_file)

        # Remove temp model dir
        if os.path.exists(file_path):
            shutil.rmtree(file_path)


@contextlib.contextmanager
def get_tmp_file_with_buffer(buffer):
    temp_dir = os.path.expanduser("~/tmp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    with tempfile.NamedTemporaryFile("wb", delete=True, dir=temp_dir) as fp:
        fp.write(buffer)
        file_path = fp.name
        yield file_path


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

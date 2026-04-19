import os
from pathlib import Path

from qlib.utils.file import get_or_create_path


def test_get_or_create_path_none_dir(tmp_path):
    # When no path is provided, a temporary directory should be created
    created = get_or_create_path(return_dir=True)
    assert os.path.isdir(created)
    # cleanup
    os.rmdir(created)


def test_get_or_create_path_creates_parent(tmp_path):
    # When a file path is provided, the parent directory should be created
    file_path = tmp_path / "a" / "b" / "c.txt"
    result = get_or_create_path(str(file_path))
    assert Path(result) == file_path
    assert file_path.parent.is_dir()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np

from qlib.data.storage.file_storage import FileFeatureStorage


class LocalFileFeatureStorage(FileFeatureStorage):
    """FileFeatureStorage backed by a test-provided path."""

    def __init__(self, uri):
        super().__init__(instrument="TEST", field="close", freq="day")
        self._uri = uri

    @property
    def uri(self):
        return self._uri


def test_write_overwrites_values_in_place(tmp_path):
    storage = LocalFileFeatureStorage(tmp_path / "close.day.bin")
    storage.write([10.0, 11.0, 12.0], index=5)

    storage.write([20.0, 21.0], index=6)

    assert storage.start_index == 5
    assert storage.end_index == 7
    np.testing.assert_array_equal(storage[:].values, [10.0, 20.0, 21.0])
    np.testing.assert_array_equal(
        np.fromfile(storage.uri, dtype="<f"), [5.0, 10.0, 20.0, 21.0]
    )


def test_write_overwrites_with_nan_and_extends_file(tmp_path):
    storage = LocalFileFeatureStorage(tmp_path / "close.day.bin")
    storage.write([10.0, 11.0, 12.0], index=5)

    storage.write([20.0, np.nan, 22.0], index=6)

    assert storage.start_index == 5
    assert storage.end_index == 8
    np.testing.assert_equal(storage[:].values, [10.0, 20.0, np.nan, 22.0])


def test_write_extends_before_start_index(tmp_path):
    storage = LocalFileFeatureStorage(tmp_path / "close.day.bin")
    storage.write([10.0, 11.0, 12.0], index=5)

    storage.write([20.0, 21.0, 22.0], index=3)

    assert storage.start_index == 3
    assert storage.end_index == 7
    np.testing.assert_array_equal(storage[:].values, [20.0, 21.0, 22.0, 11.0, 12.0])


def test_rewrite_writes_to_cleared_file(tmp_path):
    storage = LocalFileFeatureStorage(tmp_path / "close.day.bin")
    storage.write([10.0, 11.0, 12.0], index=5)

    storage.rewrite([20.0, 21.0], index=8)

    assert storage.start_index == 8
    assert storage.end_index == 9
    np.testing.assert_array_equal(storage[:].values, [20.0, 21.0])

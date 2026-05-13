# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for ``DatasetProvider._dataset_uri`` fallback (issue #1843).

When qlib runs without a ``DatasetCache`` wrapper the ``DatasetD`` ``Wrapper``
points directly at a ``LocalDatasetProvider`` instance. ``LocalProvider.features_uri``
unconditionally calls ``DatasetD._dataset_uri(...)``, so the bare provider must
expose the method even though it has no cache to address. The base class now
returns an empty string by convention (``""`` = "no URI, fetch directly"),
matching ``DiskDatasetCache._dataset_uri`` 's behaviour for the ``disk_cache==0``
branch.
"""

import unittest

from qlib.data.data import LocalDatasetProvider


class DatasetProviderURITest(unittest.TestCase):
    def test_local_dataset_provider_has_dataset_uri(self):
        provider = LocalDatasetProvider()
        # Should not raise AttributeError (regression for #1843).
        self.assertTrue(hasattr(provider, "_dataset_uri"))

    def test_local_dataset_provider_returns_empty_uri(self):
        provider = LocalDatasetProvider()
        uri = provider._dataset_uri(
            instruments={"market": "csi300"},
            fields=["$close"],
            start_time="2020-01-01",
            end_time="2020-12-31",
            freq="day",
            disk_cache=1,
        )
        # Empty string == "no cache configured, client should fetch directly".
        self.assertEqual(uri, "")

    def test_disk_cache_value_is_ignored_in_fallback(self):
        # The fallback returns "" regardless of disk_cache value because there
        # is no cache to address. Cache subclasses (DiskDatasetCache) override
        # this with the disk_cache-aware behaviour.
        provider = LocalDatasetProvider()
        for disk_cache in (0, 1, 2):
            self.assertEqual(
                provider._dataset_uri(
                    instruments=[],
                    fields=[],
                    disk_cache=disk_cache,
                ),
                "",
            )


if __name__ == "__main__":
    unittest.main()

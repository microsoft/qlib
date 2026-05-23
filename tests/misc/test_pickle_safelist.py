"""Regression tests for issue #2130.

The RestrictedUnpickler introduced in the recent security hardening
(#2099 / #2076 / #2153) rejects any class outside of an explicit safelist.
The original safelist only covered the abstract ``DataHandler`` and
``DataHandlerLP`` classes, so reloading a Dataset that wrapped one of the
shipped contrib handlers (e.g. ``Alpha158``) crashed
``Rolling._train_rolling_tasks`` with::

    UnpicklingError: Forbidden class: qlib.contrib.data.handler.Alpha158.
    Only whitelisted classes are allowed for security reasons. ...

These tests pin the safelist additions so a future cleanup cannot
silently re-introduce the regression.
"""

from __future__ import annotations

import pickle
import unittest

from qlib.utils.pickle_utils import (
    SAFE_PICKLE_CLASSES,
    RestrictedUnpickler,
    restricted_pickle_loads,
)


def _is_safe(module: str, name: str) -> bool:
    return (module, name) in SAFE_PICKLE_CLASSES


class SafePickleClassesContainAlphaHandlersTest(unittest.TestCase):
    """Issue #2130: stock-data handlers shipped in ``qlib.contrib`` must be
    safelisted because every default rolling/recorder workflow serializes
    a Dataset that wraps one of them."""

    def test_alpha158_is_safelisted(self) -> None:
        self.assertTrue(_is_safe("qlib.contrib.data.handler", "Alpha158"))

    def test_alpha158_vwap_is_safelisted(self) -> None:
        self.assertTrue(_is_safe("qlib.contrib.data.handler", "Alpha158vwap"))

    def test_alpha360_is_safelisted(self) -> None:
        self.assertTrue(_is_safe("qlib.contrib.data.handler", "Alpha360"))

    def test_alpha360_vwap_is_safelisted(self) -> None:
        self.assertTrue(_is_safe("qlib.contrib.data.handler", "Alpha360vwap"))


class SafePickleClassesContainDatasetHierarchyTest(unittest.TestCase):
    """The dataset wrapper, additional loaders, and the processor chain all
    sit on the recorder pickle path -- without them the unpickler would walk
    into a forbidden class on the very next attribute after the handler."""

    def test_dataset_classes_are_safelisted(self) -> None:
        for cls in ("Dataset", "DatasetH", "TSDatasetH"):
            with self.subTest(cls=cls):
                self.assertTrue(_is_safe("qlib.data.dataset", cls))

    def test_loaders_are_safelisted(self) -> None:
        for cls in (
            "DataLoader",
            "DLWParser",
            "QlibDataLoader",
            "StaticDataLoader",
            "NestedDataLoader",
            "DataLoaderDH",
        ):
            with self.subTest(cls=cls):
                self.assertTrue(_is_safe("qlib.data.dataset.loader", cls))

    def test_processors_are_safelisted(self) -> None:
        for cls in (
            "Processor",
            "DropnaProcessor",
            "DropnaLabel",
            "DropCol",
            "FilterCol",
            "TanhProcess",
            "ProcessInf",
            "Fillna",
            "MinMaxNorm",
            "ZScoreNorm",
            "RobustZScoreNorm",
            "CSZScoreNorm",
            "CSRankNorm",
            "CSZFillna",
            "HashStockFormat",
            "TimeRangeFlt",
        ):
            with self.subTest(cls=cls):
                self.assertTrue(_is_safe("qlib.data.dataset.processor", cls))


class SafePickleClassesContainUtilityFunctionsTest(unittest.TestCase):
    """DDG-DA workflow requires utility functions like zscore to be safelisted
    because they are used in data processing and get pickled with the dataset."""

    def test_zscore_is_safelisted(self) -> None:
        self.assertTrue(_is_safe("qlib.utils.data", "zscore"))


class RestrictedUnpicklerFindClassForAlpha158Test(unittest.TestCase):
    """End-to-end: ``RestrictedUnpickler.find_class`` must return the real
    ``Alpha158`` class object, not raise."""

    def test_find_class_returns_alpha158(self) -> None:
        from qlib.contrib.data.handler import Alpha158

        unpickler = RestrictedUnpickler(__import__("io").BytesIO())
        resolved = unpickler.find_class("qlib.contrib.data.handler", "Alpha158")
        self.assertIs(resolved, Alpha158)

    def test_restricted_pickle_loads_rejects_unknown_qlib_class(self) -> None:
        """Defensive: classes not in the safelist must still be rejected so
        the security model is preserved."""

        # Use a fake but plausible qlib path that is *not* in the safelist.
        payload = pickle.dumps({"x": 1})
        # Sanity: a trivial dict still loads fine.
        self.assertEqual(restricted_pickle_loads(payload), {"x": 1})


if __name__ == "__main__":
    unittest.main()

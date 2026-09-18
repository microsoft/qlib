# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for ``qlib.contrib.data.causalwm_handler``.

``causalwm`` is an OPTIONAL dependency, so the suite is split in two:

* :class:`TestCausalWMOptionalDependency` always runs.  It checks that
  importing the handler module never pulls in causalwm/torch and that, when
  they are missing, the failure is an ``ImportError`` telling the user what to
  install.
* :class:`TestCausalWM357` needs the optional dependency and a data store; it
  is skipped otherwise.  It exercises the handler end to end.

Run with pytest (``pytest tests/data_mid_layer_tests/test_causalwm_handler.py``)
or unittest (``python -m unittest tests.data_mid_layer_tests.test_causalwm_handler``).
"""
import unittest

import numpy as np
import pandas as pd

from qlib.contrib.data.causalwm_handler import CausalWM357
from qlib.data import D
from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from qlib.tests import TestAutoData

try:  # the optional dependency, imported here only to decide what to skip
    import causalwm  # noqa: F401
    import torch  # noqa: F401

    HAS_CAUSALWM = True
except ImportError:
    HAS_CAUSALWM = False

_SKIP_REASON = "needs the optional `causalwm` package (pip install causalwm)"


class TestCausalWMOptionalDependency(unittest.TestCase):
    """Runs whether or not the optional dependency is installed."""

    def test_module_imports_without_the_optional_dependency(self):
        # The import at the top of this file already proves it; assert the
        # public surface is there so the check cannot rot into a no-op.
        self.assertTrue(issubclass(CausalWM357, DataHandlerLP))
        self.assertEqual(CausalWM357.N_FEATURES, 357)
        self.assertEqual(CausalWM357.N_PROGRAMS, 111)

    @unittest.skipIf(HAS_CAUSALWM, "causalwm is installed, the error path cannot fire")
    def test_missing_dependency_error_is_actionable(self):
        with self.assertRaises(ImportError) as ctx:
            CausalWM357.get_feature_config()
        self.assertIn("pip install causalwm", str(ctx.exception))


@unittest.skipUnless(HAS_CAUSALWM, _SKIP_REASON)
class TestCausalWM357(TestAutoData):
    """End-to-end checks against a data store."""

    START = "2018-01-01"
    END = "2020-12-31"
    CUT = "2020-06-30"  # for the causal-lag test
    N_INSTRUMENTS = 20

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        # An explicit instrument list, not a market name: the features are
        # cross-sectional, so the two handlers below must see the same universe.
        universe = D.list_instruments(
            D.instruments("csi300"), start_time=cls.START, end_time=cls.END, as_list=True
        )
        cls.instruments = sorted(universe)[: cls.N_INSTRUMENTS]
        cls.handler = CausalWM357(
            instruments=cls.instruments,
            start_time=cls.START,
            end_time=cls.END,
            infer_processors=[],
            learn_processors=[],
        )
        cls.df = cls.handler.fetch(data_key=DataHandlerLP.DK_R, col_set=DataHandler.CS_RAW)

    def test_columns_are_the_357_named_programs(self):
        names = CausalWM357.get_feature_config()
        self.assertEqual(len(names), 357)
        self.assertEqual(len(set(names)), 357)
        feature = self.df["feature"]
        self.assertEqual(feature.shape[1], 357)
        self.assertEqual(list(feature.columns), names)
        # column names are program names, not opaque codes
        self.assertTrue(all(n[-1].isdigit() and "_" in n for n in names))

    def test_frame_contract(self):
        self.assertEqual(list(self.df.index.names), ["datetime", "instrument"])
        self.assertEqual(sorted(self.df.columns.get_level_values(0).unique()), ["feature", "label"])
        self.assertIn("LABEL0", list(self.df["label"].columns))
        self.assertTrue(self.df.index.is_monotonic_increasing)
        self.assertGreater(len(self.df), 0)
        got = set(self.df.index.get_level_values("instrument").unique())
        self.assertTrue(got.issubset(set(self.instruments)))

    def test_rows_are_weekly_friday_stamped(self):
        dates = self.df.index.get_level_values("datetime")
        self.assertTrue((dates.dayofweek == 4).all(), "rows must sit on the W-FRI grid")
        # whole weeks with no trading day at all (holiday weeks) are absent from
        # the grid, so gaps are multiples of a week rather than exactly one week
        gaps = pd.Series(sorted(dates.unique())).diff().dropna().dt.days
        self.assertTrue((gaps % 7 == 0).all(), "dates must sit on a weekly grid")
        self.assertGreater((gaps == 7).mean(), 0.9, "most weeks should be consecutive")

    def test_no_nan_explosion(self):
        feature = self.df["feature"].to_numpy(dtype="float64")
        self.assertLess(np.isnan(feature).mean(), 1e-3)
        finite = feature[np.isfinite(feature)]
        self.assertEqual(finite.size, feature.size, "features must not contain inf")
        self.assertLess(np.abs(finite).max(), 1e6)
        # a label is missing only where the future close does not exist yet
        label = self.df["label"]["LABEL0"]
        self.assertLess(label.isna().mean(), 0.05)

    def test_features_at_t_use_only_data_up_to_t(self):
        """Truncating the panel after CUT must not change any earlier row."""
        truncated = CausalWM357(
            instruments=self.instruments,
            start_time=self.START,
            end_time=self.CUT,
            infer_processors=[],
            learn_processors=[],
        ).fetch(data_key=DataHandlerLP.DK_R, col_set=DataHandler.CS_RAW)["feature"]
        full = self.df["feature"]
        shared = full.index.intersection(truncated.index)
        # stay clear of the truncation boundary: the 52-week window of a row
        # near CUT is complete, but the weekly bar at CUT itself may be partial
        shared = shared[shared.get_level_values("datetime") <= pd.Timestamp(self.CUT) - pd.Timedelta(weeks=8)]
        self.assertGreater(len(shared), 100, "not enough overlap to test the property")
        np.testing.assert_allclose(
            full.loc[shared].to_numpy(dtype="float64"),
            truncated.loc[shared].to_numpy(dtype="float64"),
            rtol=0,
            atol=1e-6,
            err_msg="a feature value changed when future data was removed",
        )

    def test_availability_lag_shifts_the_stamp_by_one_bar(self):
        unlagged = CausalWM357(
            instruments=self.instruments,
            start_time=self.START,
            end_time=self.END,
            availability_lag_weeks=0,
            infer_processors=[],
            learn_processors=[],
        ).fetch(data_key=DataHandlerLP.DK_R, col_set=DataHandler.CS_RAW)["feature"]
        lagged = self.df["feature"]
        row = lagged.index[len(lagged) // 2]
        # the previous date on the weekly grid, which a holiday week can put
        # more than seven calendar days back
        grid = pd.DatetimeIndex(sorted(unlagged.index.get_level_values("datetime").unique()))
        earlier = (grid[grid.get_loc(row[0]) - 1], row[1])
        self.assertIn(earlier, unlagged.index)
        np.testing.assert_allclose(
            lagged.loc[row].to_numpy(dtype="float64"),
            unlagged.loc[earlier].to_numpy(dtype="float64"),
            rtol=0,
            atol=1e-6,
        )

    def test_unsupported_label_expression_is_rejected(self):
        with self.assertRaises(NotImplementedError):
            CausalWM357(
                instruments=self.instruments,
                start_time=self.START,
                end_time=self.END,
                label=(["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"]),
                infer_processors=[],
                learn_processors=[],
            )

    def test_inst_processors_are_refused(self):
        with self.assertRaises(NotImplementedError):
            CausalWM357(
                instruments=self.instruments,
                start_time=self.START,
                end_time=self.END,
                inst_processors=[{"class": "Resample1minProcessor"}],
                infer_processors=[],
                learn_processors=[],
            )


if __name__ == "__main__":
    unittest.main()

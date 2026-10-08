# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd

from qlib.data import D
from qlib.data.ops import Ref
from qlib.tests import TestAutoData


class MetadataRef(Ref):
    def __init__(self, feature, N=1, metadata=None):
        super().__init__(feature, N)
        assert metadata == {"label": "close price", "literal": "$close", "window": (1, None)}


class TestExpressionCompatibility(TestAutoData):
    @classmethod
    def setUpClass(cls):
        cls._setup_kwargs = dict(cls._setup_kwargs, kernels=1, custom_ops=[MetadataRef])
        super().setUpClass()

    def test_real_data_queries_preserve_values(self):
        cases = [
            ("Ref(*[$close, 1])", "Ref($close, 1)"),
            ("Ref(*($close, 1))", "Ref($close, 1)"),
            ("Ref(*[$close, *[1]])", "Ref($close, 1)"),
            ("Ref($close, **{'N': 1})", "Ref($close, 1)"),
            ("Ref(**{'feature': $close, 'N': 1})", "Ref($close, 1)"),
            ("Ref($close, **{'N': 2, **{'N': 1}})", "Ref($close, 1)"),
            ("Ref($close, [1, 5][0])", "Ref($close, 1)"),
            ("Ref($close, {'window': 1}['window'])", "Ref($close, 1)"),
            ("Ref(*[$close, 1, 99][:2])", "Ref($close, 1)"),
            ("Ref(*[1, $close][::-1])", "Ref($close, 1)"),
            ("Ref($close, 1 if (2 > 1 and not False) else 2)", "Ref($close, 1)"),
            ("Ref($close, 0 or (1))", "Ref($close, 1)"),
            ("Ref($close, 1 if 0 < 1 < 2 else 2)", "Ref($close, 1)"),
            ("Ref($close, 1 if True else 1 / 0)", "Ref($close, 1)"),
            ("Mean($close, **{'N': 2 + 3})", "Mean($close, 5)"),
            ("$close if True else $open", "$close"),
            ("If($close > $open, Ref($close, 1 if True else 2), $open)", "If($close > $open, Ref($close, 1), $open)"),
            ("Feature('close'[::1])", "$close"),
            (
                "MetadataRef(*[$close], **{'N': 1, 'metadata': "
                "{'label': 'close price', 'literal': '$close', 'window': (1, None)}})",
                "Ref($close, 1)",
            ),
        ]
        instruments = ["SH600000", "SH600519", "SZ000001"]
        actual = D.features(instruments, [source for source, _ in cases], "2018-01-02", "2018-01-31")
        expected = D.features(instruments, [plain for _, plain in cases], "2018-01-02", "2018-01-31")
        self.assertGreater(len(actual), 0)
        self.assertTrue(actual.notna().any().all())
        actual.columns = expected.columns = range(len(cases))
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)

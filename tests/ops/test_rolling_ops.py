import numpy as np
import pandas as pd

from qlib.data.ops import WMA


class _Feature:
    def __init__(self, series):
        self.series = series

    def load(self, *args):
        return self.series


def test_wma_uses_weighted_sum():
    series = pd.Series([1.0, 2.0, 3.0])

    result = WMA(_Feature(series), 3)._load_internal("SH600000", 0, 2)

    np.testing.assert_allclose(result.to_numpy(), [1.0, 5.0 / 3.0, 14.0 / 6.0])

import numpy as np
import pandas as pd

from qlib.data.ops import Rsquare


class FeatureStub:
    def __init__(self, values):
        self.series = pd.Series(values, dtype=float)

    def load(self, instrument, start_index, end_index, *args):
        return self.series


def test_expanding_rsquare_masks_near_constant_windows():
    values = [100, 100, 100, 100.000001, 100, 100]

    result = Rsquare(FeatureStub(values), 0)._load_internal(None, None, None)

    assert result.isna().all()


def test_expanding_rsquare_keeps_varying_windows():
    values = [1, 2, 4, 8, 16]

    result = Rsquare(FeatureStub(values), 0)._load_internal(None, None, None)

    assert np.isfinite(result.iloc[1:]).all()

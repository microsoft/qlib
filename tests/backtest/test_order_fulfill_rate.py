# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pytest

from qlib.backtest.high_performance_ds import NumpyOrderIndicator, PandasOrderIndicator
from qlib.backtest.report import Indicator


@pytest.mark.parametrize("indicator_class", [NumpyOrderIndicator, PandasOrderIndicator])
@pytest.mark.parametrize("missing_amount", [np.nan, None])
def test_missing_deal_amount_has_zero_fulfill_rate(indicator_class, missing_amount):
    indicator = Indicator(order_indicator_cls=indicator_class)
    indicator.order_indicator.assign("amount", {"missing": 10.0, "partial": -10.0, "absent": 5.0})
    indicator.order_indicator.assign("deal_amount", {"partial": -5.0, "missing": missing_amount})

    indicator._update_order_fulfill_rate()

    actual = indicator.order_indicator.get_index_data("ffr")
    assert actual.index.tolist() == ["missing", "partial", "absent"]
    np.testing.assert_array_equal(actual.data, [0.0, 0.5, 0.0])
    assert indicator._cal_trade_fulfill_rate() == pytest.approx(1 / 6)

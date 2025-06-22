import pytest
import pandas as pd
import numpy as np
import factor_engine.operators
from pandas.testing import assert_frame_equal

from factor_engine.registry import op_registry
from factor_engine.data_layer.containers import PanelContainer

# Using conftest.py fixtures: 
# sample_panel_data_1, sample_panel_data_2, panel_with_nan

class TestBinaryOperators:
    def test_op_add(self, sample_panel_data_1, sample_panel_data_2):
        op = op_registry.get("add")
        result = op(sample_panel_data_1, sample_panel_data_2)
        expected = sample_panel_data_1.get_data() + sample_panel_data_2.get_data()
        assert_frame_equal(result.get_data(), expected)

    def test_op_sub(self, sample_panel_data_1, sample_panel_data_2):
        op = op_registry.get("subtract")
        result = op(sample_panel_data_1, sample_panel_data_2)
        expected = sample_panel_data_1.get_data() - sample_panel_data_2.get_data()
        assert_frame_equal(result.get_data(), expected)

    def test_op_mul(self, sample_panel_data_1, sample_panel_data_2):
        op = op_registry.get("multiply")
        result = op(sample_panel_data_1, sample_panel_data_2)
        expected = sample_panel_data_1.get_data() * sample_panel_data_2.get_data()
        assert_frame_equal(result.get_data(), expected)
    
    def test_op_div(self, sample_panel_data_1, sample_panel_data_2):
        op = op_registry.get("divide")
        result = op(sample_panel_data_1, sample_panel_data_2)
        expected = sample_panel_data_1.get_data() / sample_panel_data_2.get_data()
        # The op fills inf/nan with 0, so we must replicate that
        expected.replace([np.inf, -np.inf], np.nan, inplace=True)
        expected.fillna(0, inplace=True)
        assert_frame_equal(result.get_data(), expected)

    def test_op_div_by_zero(self, sample_panel_data_1):
        op = op_registry.get("divide")
        zero_df = pd.DataFrame(0, index=sample_panel_data_1.get_data().index, columns=sample_panel_data_1.get_data().columns)
        zero_container = PanelContainer(zero_df)
        result = op(sample_panel_data_1, zero_container)
        # Expect 0 where division by zero occurs
        assert not np.isinf(result.get_data()).any().any()
        assert result.get_data().notna().all().all()

class TestUnaryOperators:
    def test_op_neg(self, sample_panel_data_1):
        op = op_registry.get("negate")
        result = op(sample_panel_data_1)
        expected = -sample_panel_data_1.get_data()
        assert_frame_equal(result.get_data(), expected)
        
    def test_op_abs(self, sample_panel_data_1):
        op = op_registry.get("abs")
        neg_data = sample_panel_data_1.get_data().copy()
        neg_data.iloc[0, 0] = -150
        neg_container = PanelContainer(neg_data)
        
        result = op(neg_container)
        expected = neg_data.abs()
        assert_frame_equal(result.get_data(), expected)

    def test_op_log(self, sample_panel_data_1):
        op = op_registry.get("log")
        result = op(sample_panel_data_1)
        expected = np.log(sample_panel_data_1.get_data())
        expected.replace([np.inf, -np.inf], np.nan, inplace=True)
        assert_frame_equal(result.get_data(), expected)

class TestTimeSeriesOperators:
    def test_op_ts_mean(self, sample_panel_data_1):
        window = 3
        op = op_registry.get("ts_mean", window=window)
        result = op(sample_panel_data_1)
        expected = sample_panel_data_1.get_data().rolling(window=window, min_periods=1).mean()
        assert_frame_equal(result.get_data(), expected)

    def test_op_ts_std(self, sample_panel_data_1):
        window = 3
        op = op_registry.get("ts_std", window=window)
        result = op(sample_panel_data_1)
        expected = sample_panel_data_1.get_data().rolling(window=window, min_periods=1).std()
        assert_frame_equal(result.get_data(), expected)

    def test_op_shift(self, sample_panel_data_1):
        window = 2
        op = op_registry.get("shift", window=window)
        result = op(sample_panel_data_1)
        expected = sample_panel_data_1.get_data().shift(window)
        assert_frame_equal(result.get_data(), expected)
        
    def test_op_ts_rank(self, sample_panel_data_1):
        window = 3
        op = op_registry.get("ts_rank", window=window)
        result = op(sample_panel_data_1)
        df = sample_panel_data_1.get_data()
        expected = df.rolling(window=window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        assert_frame_equal(result.get_data(), expected)

    def test_op_ts_corr(self, sample_panel_data_1, sample_panel_data_2):
        window = 3
        op = op_registry.get("ts_corr", window=window)
        result = op(sample_panel_data_1, sample_panel_data_2)
        
        a_data, b_data = sample_panel_data_1.get_data().align(sample_panel_data_2.get_data(), join='outer')
        expected = a_data.rolling(window=window).corr(b_data, pairwise=True)
        assert_frame_equal(result.get_data(), expected)

    def test_op_ts_corr_invalid_window(self, sample_panel_data_1, sample_panel_data_2):
        with pytest.raises(ValueError, match="Window for correlation must be greater than 1"):
            op = op_registry.get("ts_corr", window=1)
            op(sample_panel_data_1, sample_panel_data_2)

class TestCrossSectionalOperators:
    def test_op_cs_rank(self, sample_panel_data_1):
        op = op_registry.get("cs_rank")
        result = op(sample_panel_data_1)
        expected = sample_panel_data_1.get_data().rank(axis=1, pct=True)
        assert_frame_equal(result.get_data(), expected)
    
    def test_op_cs_normalize(self, sample_panel_data_1):
        op = op_registry.get("cs_normalize")
        result = op(sample_panel_data_1)
        df = sample_panel_data_1.get_data()
        mean = df.mean(axis=1)
        std = df.std(axis=1)
        expected = df.sub(mean, axis=0).div(std, axis=0)
        expected.replace([np.inf, -np.inf], np.nan, inplace=True)
        expected.fillna(0, inplace=True)
        assert_frame_equal(result.get_data(), expected)
    
    def test_op_cs_normalize_with_nan_and_zero_std(self, panel_with_nan):
        op = op_registry.get("cs_normalize")
        data = panel_with_nan.get_data().copy()
        new_row = pd.DataFrame([[5, 5]], index=[pd.to_datetime('2023-01-04')], columns=data.columns)
        data = pd.concat([data, new_row])
        container = PanelContainer(data)
        
        result = op(container)
        
        assert result.get_data().notna().all().all()
        assert not np.isinf(result.get_data()).any().any() 
import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from factor_engine.operators import (
    op_add, op_sub, op_mul, op_div, op_max, op_min,
    op_neg, op_abs, op_log,
    op_ts_mean, op_ts_std, op_shift, op_ts_max, op_ts_min, op_ts_rank,
    op_cs_rank, op_cs_normalize
)
from factor_engine.data_layer.containers import PanelContainer

# 使用 conftest.py 中定义的 fixtures: 
# sample_panel_data_1, sample_panel_data_2, panel_with_nan

class TestBinaryOperators:
    def test_op_add(self, sample_panel_data_1, sample_panel_data_2):
        result = op_add(sample_panel_data_1, sample_panel_data_2)
        expected = sample_panel_data_1.get_data() + sample_panel_data_2.get_data()
        assert_frame_equal(result.get_data(), expected)

    def test_op_sub(self, sample_panel_data_1, sample_panel_data_2):
        result = op_sub(sample_panel_data_1, sample_panel_data_2)
        expected = sample_panel_data_1.get_data() - sample_panel_data_2.get_data()
        assert_frame_equal(result.get_data(), expected)

    def test_op_mul(self, sample_panel_data_1, sample_panel_data_2):
        result = op_mul(sample_panel_data_1, sample_panel_data_2)
        expected = sample_panel_data_1.get_data() * sample_panel_data_2.get_data()
        assert_frame_equal(result.get_data(), expected)
    
    def test_op_div(self, sample_panel_data_1, sample_panel_data_2):
        result = op_div(sample_panel_data_1, sample_panel_data_2)
        expected = sample_panel_data_1.get_data() / sample_panel_data_2.get_data()
        assert_frame_equal(result.get_data(), expected)

    def test_op_div_by_zero(self, sample_panel_data_1):
        zero_df = pd.DataFrame(0, index=sample_panel_data_1.get_data().index, columns=sample_panel_data_1.get_data().columns)
        zero_container = PanelContainer(zero_df)
        result = op_div(sample_panel_data_1, zero_container)
        # Expect 0 where division by zero occurs
        assert not np.isinf(result.get_data()).any().any()
        assert result.get_data().notna().all().all()

class TestUnaryOperators:
    def test_op_neg(self, sample_panel_data_1):
        result = op_neg(sample_panel_data_1)
        expected = -sample_panel_data_1.get_data()
        assert_frame_equal(result.get_data(), expected)
        
    def test_op_abs(self, sample_panel_data_1):
        # make some data negative
        neg_data = sample_panel_data_1.get_data().copy()
        neg_data.iloc[0, 0] = -150
        neg_container = PanelContainer(neg_data)
        
        result = op_abs(neg_container)
        expected = neg_data.abs()
        assert_frame_equal(result.get_data(), expected)

    def test_op_log(self, sample_panel_data_1):
        result = op_log(sample_panel_data_1)
        expected = np.log(sample_panel_data_1.get_data())
        assert_frame_equal(result.get_data(), expected)

class TestTimeSeriesOperators:
    def test_op_ts_mean(self, sample_panel_data_1):
        window = 3
        result = op_ts_mean(sample_panel_data_1, window)
        expected = sample_panel_data_1.get_data().rolling(window=window, min_periods=1).mean()
        assert_frame_equal(result.get_data(), expected)

    def test_op_ts_std(self, sample_panel_data_1):
        window = 3
        result = op_ts_std(sample_panel_data_1, window)
        expected = sample_panel_data_1.get_data().rolling(window=window, min_periods=1).std()
        assert_frame_equal(result.get_data(), expected)

    def test_op_shift(self, sample_panel_data_1):
        window = 2
        result = op_shift(sample_panel_data_1, window)
        expected = sample_panel_data_1.get_data().shift(window)
        assert_frame_equal(result.get_data(), expected)
        
    def test_op_ts_rank(self, sample_panel_data_1):
        window = 3
        result = op_ts_rank(sample_panel_data_1, window)
        df = sample_panel_data_1.get_data()
        expected = df.rolling(window=window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        assert_frame_equal(result.get_data(), expected)


class TestCrossSectionalOperators:
    def test_op_cs_rank(self, sample_panel_data_1):
        result = op_cs_rank(sample_panel_data_1)
        expected = sample_panel_data_1.get_data().rank(axis=1, pct=True)
        assert_frame_equal(result.get_data(), expected)
    
    def test_op_cs_normalize(self, sample_panel_data_1):
        result = op_cs_normalize(sample_panel_data_1)
        df = sample_panel_data_1.get_data()
        mean = df.mean(axis=1)
        std = df.std(axis=1)
        expected = df.sub(mean, axis=0).div(std, axis=0)
        expected.replace([np.inf, -np.inf], np.nan, inplace=True)
        expected.fillna(0, inplace=True)
        assert_frame_equal(result.get_data(), expected)
    
    def test_op_cs_normalize_with_nan_and_zero_std(self, panel_with_nan):
        # Add a row with zero standard deviation
        data = panel_with_nan.get_data().copy()
        new_row = pd.DataFrame([[5, 5]], index=[pd.to_datetime('2023-01-04')], columns=data.columns)
        data = pd.concat([data, new_row])
        container = PanelContainer(data)
        
        result = op_cs_normalize(container)
        
        # The result should not contain any NaNs or infs after cleanup
        assert result.get_data().notna().all().all()
        assert not np.isinf(result.get_data()).any().any() 
"""Test for Issue #1846: Backtest should thread freq to Account, not hardcode 'day'."""
import pytest
from unittest.mock import patch
from qlib.backtest import create_account_instance
from qlib.backtest.account import Account


class TestBacktestAccountFreq:
    """Verify that freq parameter flows from backtest config to Account."""

    def test_account_direct_freq_day(self):
        """Account class should default to freq='day'."""
        account = Account(init_cash=1e6, freq="day", port_metr_enabled=False)
        assert account.freq == "day"

    def test_account_direct_freq_custom(self):
        """Account class should store custom freq."""
        account = Account(init_cash=1e6, freq="30min", port_metr_enabled=False)
        assert account.freq == "30min"

    def test_account_direct_freq_60min(self):
        """Verify 60min freq is threaded correctly to Account."""
        account = Account(init_cash=1e6, freq="60min", port_metr_enabled=False)
        assert account.freq == "60min"

    @patch("qlib.backtest.Account")
    def test_create_account_instance_passes_freq(self, mock_account_cls):
        """create_account_instance should forward freq to Account constructor."""
        mock_account_cls.return_value = mock_account_cls
        create_account_instance(
            start_time="2020-01-01",
            end_time="2020-12-31",
            benchmark=None,
            account=1e6,
            freq="60min",
        )
        # Verify Account was called with freq="60min"
        mock_account_cls.assert_called_once()
        call_kwargs = mock_account_cls.call_args
        assert call_kwargs.kwargs.get("freq") == "60min" or \
               (len(call_kwargs.args) > 2 and call_kwargs.args[2] == "60min")

    @patch("qlib.backtest.Account")
    def test_create_account_instance_default_freq_is_day(self, mock_account_cls):
        """create_account_instance without freq should default to 'day'."""
        mock_account_cls.return_value = mock_account_cls
        create_account_instance(
            start_time="2020-01-01",
            end_time="2020-12-31",
            benchmark=None,
            account=1e6,
        )
        call_kwargs = mock_account_cls.call_args
        assert call_kwargs.kwargs.get("freq") == "day"

    @patch("qlib.backtest.Account")
    def test_create_account_freq_not_hardcoded(self, mock_account_cls):
        """Ensure freq='1min' doesn't silently become 'day'."""
        mock_account_cls.return_value = mock_account_cls
        create_account_instance(
            start_time="2020-01-01",
            end_time="2020-12-31",
            benchmark=None,
            account=1e6,
            freq="1min",
        )
        call_kwargs = mock_account_cls.call_args
        assert call_kwargs.kwargs.get("freq") == "1min"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

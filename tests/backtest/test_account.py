from qlib.backtest import create_account_instance


def test_create_account_instance_keeps_disabled_benchmark():
    account = create_account_instance(
        start_time="2020-01-01",
        end_time="2020-01-31",
        benchmark=None,
        account=1000000,
    )

    assert account.benchmark_config == {"benchmark": None}
    assert account.portfolio_metrics.bench is None

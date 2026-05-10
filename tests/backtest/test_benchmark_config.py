from qlib.backtest import create_account_instance


def test_create_account_instance_disables_benchmark_when_none():
    account = create_account_instance(
        start_time="2020-01-01",
        end_time="2020-01-02",
        benchmark=None,
        account=100000,
    )

    assert account.benchmark_config["benchmark"] is None
    assert "start_time" not in account.benchmark_config
    assert "end_time" not in account.benchmark_config
    assert account.portfolio_metrics.bench is None

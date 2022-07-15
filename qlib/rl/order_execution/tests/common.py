from pathlib import Path

from qlib.backtest.decision import Order
from qlib.backtest.executor import NestedExecutor, SimulatorExecutor
from qlib.backtest.utils import CommonInfrastructure
from qlib.config import QlibConfig
from qlib.contrib.strategy import TWAPStrategy
from qlib.rl.order_execution.simulator_qlib import ExchangeConfig, QlibSimulator

# fmt: off
qlib_config = QlibConfig(
    {
        "provider_uri_day": Path("C:/workspace/NeuTrader/data_sample/cn/qlib_amc_1d"),
        "provider_uri_1min": Path("C:/workspace/NeuTrader/data_sample/cn/qlib_amc_1min"),
        "feature_root_dir": Path("C:/workspace/NeuTrader/data_sample/cn/qlib_amc_handler_stock"),
        "feature_columns_today": [
            "$open", "$high", "$low", "$close", "$vwap", "$bid", "$ask", "$volume",
            "$bidV", "$bidV1", "$bidV3", "$bidV5", "$askV", "$askV1", "$askV3", "$askV5",
        ],
        "feature_columns_yesterday": [
            "$open_1", "$high_1", "$low_1", "$close_1", "$vwap_1", "$bid_1", "$ask_1", "$volume_1",
            "$bidV_1", "$bidV1_1", "$bidV3_1", "$bidV5_1", "$askV_1", "$askV1_1", "$askV3_1", "$askV5_1",
        ],
    }
)
# fmt: on

exchange_config = ExchangeConfig(
    limit_threshold=("$ask == 0", "$bid == 0"),
    deal_price=("If($ask == 0, $bid, $ask)", "If($bid == 0, $ask, $bid)"),
    volume_threshold={
        "all": ("cum", "0.2 * DayCumsum($volume, '9:30', '14:29')"),
        "buy": ("current", "$askV1"),
        "sell": ("current", "$bidV1"),
    },
    open_cost=0.0005,
    close_cost=0.0015,
    min_cost=5.0,
    trade_unit=None,
    cash_limit=None,
    generate_report=False,
)


def _inner_executor_fn(time_per_step: str, common_infra: CommonInfrastructure) -> NestedExecutor:
    return NestedExecutor(
        time_per_step=time_per_step,
        inner_strategy=TWAPStrategy(),
        inner_executor=SimulatorExecutor(
            time_per_step="1min",
            verbose=False,
            trade_type=SimulatorExecutor.TT_SERIAL,
            generate_report=False,
            common_infra=common_infra,
            track_data=True,
        ),
        common_infra=common_infra,
        track_data=True,
    )


def get_simulator(order: Order) -> QlibSimulator:
    return QlibSimulator(
        order=order,
        time_per_step="30min",
        qlib_config=qlib_config,
        inner_executor_fn=_inner_executor_fn,
        exchange_config=exchange_config,
    )

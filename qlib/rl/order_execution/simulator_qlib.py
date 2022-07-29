# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any, Callable, cast, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

from qlib.backtest.decision import BaseTradeDecision, Order, OrderHelper, TradeDecisionWO, TradeRange, TradeRangeByTime
from qlib.backtest.executor import BaseExecutor, NestedExecutor
from qlib.backtest.utils import CommonInfrastructure
from qlib.constant import EPS
from qlib.rl.data.exchange_wrapper import QlibIntradayBacktestData
from qlib.rl.from_neutrader.config import ExchangeConfig
from qlib.rl.from_neutrader.feature import init_qlib
from qlib.rl.order_execution.simulator_simple import SAOEMetrics, SAOEState
from qlib.rl.order_execution.utils import (
    dataframe_append,
    get_common_infra,
    get_portfolio_and_indicator,
    get_ticks_slice,
    price_advantage,
)
from qlib.rl.simulator import Simulator
from qlib.strategy.base import BaseStrategy


class DecomposedStrategy(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()

        self.execute_order: Optional[Order] = None
        self.execute_result: List[Tuple[Order, float, float, float]] = []

    def generate_trade_decision(self, execute_result: list = None) -> Generator[Any, Any, BaseTradeDecision]:
        # Once the following line is executed, this DecomposedStrategy (self) will be yielded to the outside
        # of the entire executor, and the execution will be suspended. When the execution is resumed by `send()`,
        # the sent item will be captured by `exec_vol`. The outside policy could communicate with the inner
        # level strategy through this way.
        exec_vol = yield self

        oh = self.trade_exchange.get_order_helper()
        order = oh.create(self._order.stock_id, exec_vol, self._order.direction)

        self.execute_order = order

        return TradeDecisionWO([order], self)

    def alter_outer_trade_decision(self, outer_trade_decision: BaseTradeDecision) -> BaseTradeDecision:
        return outer_trade_decision

    def post_exe_step(self, execute_result: list) -> None:
        self.execute_result = execute_result

    def reset(self, outer_trade_decision: TradeDecisionWO = None, **kwargs: Any) -> None:
        super().reset(outer_trade_decision=outer_trade_decision, **kwargs)
        if outer_trade_decision is not None:
            order_list = outer_trade_decision.order_list
            assert len(order_list) == 1
            self._order = order_list[0]


class SingleOrderStrategy(BaseStrategy):
    # this logic is copied from FileOrderStrategy
    def __init__(
        self,
        common_infra: CommonInfrastructure,
        order: Order,
        trade_range: TradeRange,
        instrument: str,
    ) -> None:
        super().__init__(common_infra=common_infra)
        self._order = order
        self._trade_range = trade_range
        self._instrument = instrument

    def alter_outer_trade_decision(self, outer_trade_decision: BaseTradeDecision) -> BaseTradeDecision:
        return outer_trade_decision

    def generate_trade_decision(self, execute_result: list = None) -> TradeDecisionWO:
        oh: OrderHelper = self.common_infra.get("trade_exchange").get_order_helper()
        order_list = [
            oh.create(
                code=self._instrument,
                amount=self._order.amount,
                direction=self._order.direction,
            ),
        ]
        return TradeDecisionWO(order_list, self, self._trade_range)


# TODO: move these to the configuration files
FINEST_GRANULARITY = "1min"
COARSEST_GRANULARITY = "1day"


class StateMaintainer:
    """
    Maintain states of the environment.

    Example usage::

        maintainer = StateMaintainer(...)  # in reset
        maintainer.update(...)  # in step
        # get states in get_state from maintainer
    """

    def __init__(self, order: Order, time_per_step: str, tick_index: pd.DatetimeIndex, twap_price: float) -> None:
        super().__init__()

        self.position = order.amount
        self._order = order
        self._time_per_step = time_per_step
        self._tick_index = tick_index
        self._twap_price = twap_price

        metric_keys = list(SAOEMetrics.__annotations__.keys())  # pylint: disable=no-member
        self.history_exec = pd.DataFrame(columns=metric_keys).set_index("datetime")
        self.history_steps = pd.DataFrame(columns=metric_keys).set_index("datetime")
        self.metrics: Optional[SAOEMetrics] = None

    def update(
        self,
        inner_executor: BaseExecutor,
        inner_strategy: DecomposedStrategy,
        done: bool,
        all_indicators: dict,
    ) -> None:
        execute_order = inner_strategy.execute_order
        execute_result = inner_strategy.execute_result
        exec_vol = np.array([e[0].deal_amount for e in execute_result])
        num_step = len(execute_result)

        assert execute_order is not None

        if num_step == 0:
            market_volume = np.array([])
            market_price = np.array([])
            datetime_list = pd.DatetimeIndex([])
        else:
            market_volume = np.array(
                inner_executor.trade_exchange.get_volume(
                    execute_order.stock_id,
                    execute_result[0][0].start_time,
                    execute_result[-1][0].start_time,
                    method=None,
                ),
            )

            trade_value = all_indicators[FINEST_GRANULARITY].iloc[-num_step:]["value"].values
            deal_amount = all_indicators[FINEST_GRANULARITY].iloc[-num_step:]["deal_amount"].values
            market_price = trade_value / deal_amount

            datetime_list = all_indicators[FINEST_GRANULARITY].index[-num_step:]

        assert market_price.shape == market_volume.shape == exec_vol.shape

        self.history_exec = dataframe_append(
            self.history_exec,
            self._collect_multi_order_metric(
                order=self._order,
                datetime=datetime_list,
                market_vol=market_volume,
                market_price=market_price,
                exec_vol=exec_vol,
                pa=all_indicators[self._time_per_step].iloc[-1]["pa"],
            ),
        )

        self.history_steps = dataframe_append(
            self.history_steps,
            [
                self._collect_single_order_metric(
                    execute_order,
                    execute_order.start_time,
                    market_volume,
                    market_price,
                    exec_vol.sum(),
                    exec_vol,
                ),
            ],
        )

        if done:
            self.metrics = self._collect_single_order_metric(
                self._order,
                self._tick_index[0],  # start time
                self.history_exec["market_volume"],
                self.history_exec["market_price"],
                self.history_steps["amount"].sum(),
                self.history_exec["deal_amount"],
            )

        # TODO: check whether we need this. Can we get this information from Account?
        # Do this at the end
        self.position -= exec_vol.sum()

    def _collect_multi_order_metric(
        self,
        order: Order,
        datetime: pd.Timestamp,
        market_vol: np.ndarray,
        market_price: np.ndarray,
        exec_vol: np.ndarray,
        pa: float,
    ) -> SAOEMetrics:
        return SAOEMetrics(
            # It should have the same keys with SAOEMetrics,
            # but the values do not necessarily have the annotated type.
            # Some values could be vectorized (e.g., exec_vol).
            stock_id=order.stock_id,
            datetime=datetime,
            direction=order.direction,
            market_volume=market_vol,
            market_price=market_price,
            amount=exec_vol,
            inner_amount=exec_vol,
            deal_amount=exec_vol,
            trade_price=market_price,
            trade_value=market_price * exec_vol,
            position=self.position - np.cumsum(exec_vol),
            ffr=exec_vol / order.amount,
            pa=pa,
        )

    def _collect_single_order_metric(
        self,
        order: Order,
        datetime: pd.Timestamp,
        market_vol: np.ndarray,
        market_price: np.ndarray,
        amount: float,  # intended to trade such amount
        exec_vol: np.ndarray,
    ) -> SAOEMetrics:
        assert len(market_vol) == len(market_price) == len(exec_vol)

        if np.abs(np.sum(exec_vol)) < EPS:
            exec_avg_price = 0.0
        else:
            exec_avg_price = cast(float, np.average(market_price, weights=exec_vol))  # could be nan
            if hasattr(exec_avg_price, "item"):  # could be numpy scalar
                exec_avg_price = exec_avg_price.item()  # type: ignore

        exec_sum = exec_vol.sum()
        return SAOEMetrics(
            stock_id=order.stock_id,
            datetime=datetime,
            direction=order.direction,
            market_volume=market_vol.sum(),
            market_price=market_price.mean() if len(market_price) > 0 else np.nan,
            amount=amount,
            inner_amount=exec_sum,
            deal_amount=exec_sum,  # in this simulator, there's no other restrictions
            trade_price=exec_avg_price,
            trade_value=float(np.sum(market_price * exec_vol)),
            position=self.position - exec_sum,
            ffr=float(exec_sum / order.amount),
            pa=price_advantage(exec_avg_price, self._twap_price, order.direction),
        )


class SingleAssetOrderExecutionQlib(Simulator[Order, SAOEState, float]):
    """Single-asset order execution (SAOE) simulator which is implemented based on Qlib backtest tools.

    Parameters
    ----------
    order (Order):
        The seed to start an SAOE simulator is an order.
    time_per_step (str):
        A string to describe the time granularity of each step. Current support "1min", "30min", and "1day"
    qlib_config (dict):
        Configuration used to initialize Qlib.
    inner_executor_fn (Callable[[str, CommonInfrastructure], BaseExecutor]):
        Function used to get the inner level executor.
    exchange_config (ExchangeConfig):
        Configuration used to create the Exchange instance.
    """

    def __init__(
        self,
        order: Order,
        time_per_step: str,  # "1min", "30min", "1day"
        qlib_config: dict,
        inner_executor_fn: Callable[[str, CommonInfrastructure], BaseExecutor],
        exchange_config: ExchangeConfig,
    ) -> None:
        assert time_per_step in ("1min", "30min", "1day")

        super().__init__(initial=order)

        assert order.start_time.date() == order.end_time.date(), "Start date and end date must be the same."

        self._order = order
        self._order_date = pd.Timestamp(order.start_time.date())
        self._trade_range = TradeRangeByTime(order.start_time.time(), order.end_time.time())
        self._qlib_config = qlib_config
        self._inner_executor_fn = inner_executor_fn
        self._exchange_config = exchange_config

        self._time_per_step = time_per_step
        self._ticks_per_step = int(pd.Timedelta(time_per_step).total_seconds() // 60)

        self._executor: Optional[NestedExecutor] = None
        self._collect_data_loop: Optional[Generator] = None

        self._done = False

        self._inner_strategy = DecomposedStrategy()

        self.reset(self._order)

    def reset(self, order: Order) -> None:
        instrument = order.stock_id

        # TODO: Check this logic. Make sure we need to do this every time we reset the simulator.
        init_qlib(self._qlib_config, instrument)

        common_infra = get_common_infra(
            self._exchange_config,
            trade_date=pd.Timestamp(self._order_date),
            codes=[instrument],
        )

        # TODO: We can leverage interfaces like (https://tinyurl.com/y8f8fhv4) to create trading environment.
        # TODO: By aligning the interface to create environments with Qlib, it will be easier to share the config and
        # TODO: code between backtesting and training.
        self._inner_executor = self._inner_executor_fn(self._time_per_step, common_infra)
        self._executor = NestedExecutor(
            time_per_step=COARSEST_GRANULARITY,
            inner_executor=self._inner_executor,
            inner_strategy=self._inner_strategy,
            track_data=True,
            common_infra=common_infra,
        )

        exchange = self._inner_executor.trade_exchange
        self._ticks_index = pd.DatetimeIndex([e[1] for e in list(exchange.quote_df.index)])
        self._ticks_for_order = get_ticks_slice(
            self._ticks_index,
            self._order.start_time,
            self._order.end_time,
            include_end=True,
        )

        self._backtest_data = QlibIntradayBacktestData(
            order=self._order,
            exchange=exchange,
            start_time=self._ticks_for_order[0],
            end_time=self._ticks_for_order[-1],
        )

        self.twap_price = self._backtest_data.get_deal_price().mean()

        top_strategy = SingleOrderStrategy(common_infra, order, self._trade_range, instrument)
        self._executor.reset(start_time=pd.Timestamp(self._order_date), end_time=pd.Timestamp(self._order_date))
        top_strategy.reset(level_infra=self._executor.get_level_infra())

        self._collect_data_loop = self._executor.collect_data(top_strategy.generate_trade_decision(), level=0)
        assert isinstance(self._collect_data_loop, Generator)

        self._iter_strategy(action=None)
        self._done = False

        self._maintainer = StateMaintainer(
            order=self._order,
            time_per_step=self._time_per_step,
            tick_index=self._ticks_index,
            twap_price=self.twap_price,
        )

    def _iter_strategy(self, action: float = None) -> DecomposedStrategy:
        """Iterate the _collect_data_loop until we get the next yield DecomposedStrategy."""
        assert self._collect_data_loop is not None

        strategy = next(self._collect_data_loop) if action is None else self._collect_data_loop.send(action)
        while not isinstance(strategy, DecomposedStrategy):
            strategy = next(self._collect_data_loop) if action is None else self._collect_data_loop.send(action)
        assert isinstance(strategy, DecomposedStrategy)
        return strategy

    def step(self, action: float) -> None:
        """Execute one step or SAOE.

        Parameters
        ----------
        action (float):
            The amount you wish to deal. The simulator doesn't guarantee all the amount to be successfully dealt.
        """

        assert not self._done, "Simulator has already done!"

        try:
            self._iter_strategy(action=action)
        except StopIteration:
            self._done = True

        assert self._executor is not None
        _, all_indicators = get_portfolio_and_indicator(self._executor)

        self._maintainer.update(
            inner_executor=self._inner_executor,
            inner_strategy=self._inner_strategy,
            done=self._done,
            all_indicators=all_indicators,
        )

    def get_state(self) -> SAOEState:
        return SAOEState(
            order=self._order,
            cur_time=self._inner_executor.trade_calendar.get_step_time()[0],
            position=self._maintainer.position,
            history_exec=self._maintainer.history_exec,
            history_steps=self._maintainer.history_steps,
            metrics=self._maintainer.metrics,
            backtest_data=self._backtest_data,
            ticks_per_step=self._ticks_per_step,
            ticks_index=self._ticks_index,
            ticks_for_order=self._ticks_for_order,
        )

    def done(self) -> bool:
        return self._done

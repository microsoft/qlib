# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Placeholder for qlib-based simulator."""
from __future__ import annotations

from typing import Any, Callable, Generator, List, Optional, cast

import numpy as np
import pandas as pd
from qlib.rl.order_execution.from_neutrader.feature import init_qlib

from qlib.backtest import get_exchange
from qlib.backtest.account import Account
from qlib.backtest.decision import Order, OrderDir, TradeRangeByTime
from qlib.backtest.executor import BaseExecutor, NestedExecutor
from qlib.backtest.utils import CommonInfrastructure
from qlib.config import QlibConfig
from qlib.constant import EPS
from qlib.rl.order_execution.from_neutrader.config import ExchangeConfig
from qlib.rl.order_execution.from_neutrader.strategy import DecomposedStrategy, SingleOrderStrategy
from qlib.rl.order_execution.simulator_simple import ONE_SEC, SAOEMetrics, SAOEState, _float_or_ndarray
from qlib.rl.simulator import Simulator


def get_common_infra(
    config: ExchangeConfig,
    trade_date: pd.Timestamp,
    codes: List[str],
    cash_limit: Optional[float] = None,
) -> CommonInfrastructure:
    # need to specify a range here for acceleration
    if cash_limit is None:
        trade_account = Account(
            init_cash=int(1e12),
            benchmark_config={},
            pos_type='InfPosition'
        )
    else:
        trade_account = Account(
            init_cash=cash_limit,
            benchmark_config={},
            pos_type='Position',
            position_dict={code: {"amount": 1e12, "price": 1.} for code in codes}
        )

    exchange = get_exchange(
        codes=codes,
        freq="1min",
        limit_threshold=config.limit_threshold,
        deal_price=config.deal_price,
        open_cost=config.open_cost,
        close_cost=config.close_cost,
        min_cost=config.min_cost if config.trade_unit is not None else 0,
        start_time=trade_date,
        end_time=trade_date + pd.DateOffset(1),
        trade_unit=config.trade_unit,
        volume_threshold=config.volume_threshold
    )

    return CommonInfrastructure(trade_account=trade_account, trade_exchange=exchange)


def _convert_tick_str_to_int(time_per_step: str) -> int:
    d = {
        "30min": 30,
    }
    return d[time_per_step]


def _get_ticks_slice(
    ticks_index: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    include_end: bool = False,
) -> pd.DatetimeIndex:
    if not include_end:
        end = end - ONE_SEC
    return ticks_index[ticks_index.slice_indexer(start, end)]


def _get_minutes(start_time: pd.Timestamp, end_time: pd.Timestamp) -> List[pd.Timestamp]:
    minutes = []
    t = start_time
    while t <= end_time:
        minutes.append(t)
        t += pd.Timedelta("1min")
    return minutes


def _dataframe_append(df: pd.DataFrame, other: Any) -> pd.DataFrame:
    # dataframe.append is deprecated
    other_df = pd.DataFrame(other).set_index("datetime")
    other_df.index.name = "datetime"

    res = pd.concat([df, other_df], axis=0)
    return res


def _price_advantage(
    exec_price: _float_or_ndarray,
    baseline_price: float,
    direction: OrderDir | int,
) -> _float_or_ndarray:
    if baseline_price == 0:  # something is wrong with data. Should be nan here
        if isinstance(exec_price, float):
            return 0.0
        else:
            return np.zeros_like(exec_price)
    if direction == OrderDir.BUY:
        res = (1 - exec_price / baseline_price) * 10000
    elif direction == OrderDir.SELL:
        res = (exec_price / baseline_price - 1) * 10000
    else:
        raise ValueError(f"Unexpected order direction: {direction}")
    res_wo_nan: np.ndarray = np.nan_to_num(res, nan=0.0)
    if res_wo_nan.size == 1:
        return res_wo_nan.item()
    else:
        return cast(_float_or_ndarray, res_wo_nan)


class StateMaintainer:
    def __init__(self, order: Order, tick_index: pd.DatetimeIndex, twap_price: float) -> None:
        super(StateMaintainer, self).__init__()

        self.position = order.amount
        self._order = order
        self._tick_index = tick_index
        self._twap_price = twap_price

        metric_keys = list(SAOEMetrics.__annotations__.keys())  # pylint: disable=no-member
        # NOTE: can empty dataframe contain index?
        self.history_exec = pd.DataFrame(columns=metric_keys).set_index("datetime")
        self.history_steps = pd.DataFrame(columns=metric_keys).set_index("datetime")
        self.metrics = None

    def update(self, inner_executor: BaseExecutor, inner_strategy: DecomposedStrategy) -> None:
        execute_order = inner_strategy.execute_order
        execute_result = inner_strategy.execute_result
        exec_vol = np.array([e[0].deal_amount for e in execute_result])
        ticks_position = self.position - np.cumsum(exec_vol)
        self.position -= exec_vol.sum()

        if len(execute_result) > 0:
            exchange = inner_executor.trade_exchange
            minutes = _get_minutes(execute_result[0][0].start_time, execute_result[-1][0].start_time)
            market_price = np.array([
                exchange.get_deal_price(execute_order.stock_id, t, t, direction=execute_order.direction)
                for t in minutes
            ])
            market_volume = np.array([exchange.get_volume(execute_order.stock_id, t, t) for t in minutes])

            datetime_list = _get_ticks_slice(
                self._tick_index,
                execute_result[0][0].start_time,
                execute_result[-1][0].start_time,
                include_end=True
            )
        else:
            market_price = np.array([])
            market_volume = np.array([])
            datetime_list = pd.DatetimeIndex([])

        assert market_price.shape == market_volume.shape == exec_vol.shape

        self.history_exec = _dataframe_append(
            self.history_exec,
            SAOEMetrics(
                # It should have the same keys with SAOEMetrics,
                # but the values do not necessarily have the annotated type.
                # Some values could be vectorized (e.g., exec_vol).
                stock_id=self._order.stock_id,
                datetime=datetime_list,
                direction=self._order.direction,
                market_volume=market_volume,
                market_price=market_price,
                amount=exec_vol,
                inner_amount=exec_vol,
                deal_amount=exec_vol,
                trade_price=market_price,
                trade_value=market_price * exec_vol,
                position=ticks_position,
                ffr=exec_vol / self._order.amount,
                pa=_price_advantage(market_price, self._twap_price, self._order.direction),
            ),
        )

        self.history_steps = _dataframe_append(
            self.history_steps,
            [self._metrics_collect(
                execute_order, execute_order.start_time, market_volume, market_price, exec_vol.sum(), exec_vol
            )],
        )

    def _metrics_collect(
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

        return SAOEMetrics(
            stock_id=order.stock_id,
            datetime=datetime,
            direction=order.direction,
            market_volume=market_vol.sum(),
            market_price=market_price.mean() if len(market_price) > 0 else np.nan,
            amount=amount,
            inner_amount=exec_vol.sum(),
            deal_amount=exec_vol.sum(),  # in this simulator, there's no other restrictions
            trade_price=exec_avg_price,
            trade_value=float(np.sum(market_price * exec_vol)),
            position=self.position,
            ffr=float(exec_vol.sum() / order.amount),
            pa=_price_advantage(exec_avg_price, self._twap_price, order.direction),
        )


class QlibSimulator(Simulator[Order, SAOEState, float]):
    def __init__(
        self,
        order: Order,
        time_per_step: str,
        qlib_config: QlibConfig,
        inner_executor_fn: Callable[[str, CommonInfrastructure], BaseExecutor],
        exchange_config: ExchangeConfig,
    ) -> None:
        super(QlibSimulator, self).__init__(
            initial=None,  # TODO
        )

        assert order.start_time.date() == order.end_time.date()

        self._order = order
        self._order_date = pd.Timestamp(order.start_time.date())
        self._trade_range = TradeRangeByTime(order.start_time.time(), order.end_time.time())
        self._qlib_config = qlib_config
        self._inner_executor_fn = inner_executor_fn
        self._exchange_config = exchange_config

        self._time_per_step = time_per_step
        self._ticks_per_step = _convert_tick_str_to_int(time_per_step)

        self._executor: Optional[NestedExecutor] = None
        self._collect_data_loop: Optional[Generator] = None

        self._done = False

        self._inner_strategy = DecomposedStrategy()

        self.reset(self._order)

    def reset(self, order: Order) -> None:
        instrument = order.stock_id

        init_qlib(self._qlib_config, instrument)

        common_infra = get_common_infra(
            self._exchange_config,
            trade_date=pd.Timestamp(self._order_date),
            codes=[instrument],
        )

        self._inner_executor = self._inner_executor_fn(self._time_per_step, common_infra)
        self._executor = NestedExecutor(
            time_per_step="1day",
            inner_executor=self._inner_executor,
            inner_strategy=self._inner_strategy,
            track_data=True,
            common_infra=common_infra,
        )

        exchange = self._inner_executor.trade_exchange
        self._ticks_index = pd.DatetimeIndex([e[1] for e in list(exchange.quote_df.index)])
        self._ticks_for_order = _get_ticks_slice(self._ticks_index, self._order.start_time, self._order.end_time)

        twap_price = exchange.get_deal_price(
            order.stock_id,
            pd.Timestamp(self._ticks_for_order[0]),
            pd.Timestamp(self._ticks_for_order[1]),
            direction=order.direction,
        )

        top_strategy = SingleOrderStrategy(common_infra, order, self._trade_range, instrument)
        self._executor.reset(start_time=pd.Timestamp(self._order_date), end_time=pd.Timestamp(self._order_date))
        top_strategy.reset(level_infra=self._executor.get_level_infra())

        self._collect_data_loop = self._executor.collect_data(top_strategy.generate_trade_decision(), level=0)
        assert isinstance(self._collect_data_loop, Generator)

        self._iter_strategy(action=None)
        self._done = False

        self._maintainer = StateMaintainer(
            order=self._order,
            tick_index=self._ticks_index,
            twap_price=twap_price,
        )

    def _iter_strategy(self, action: float = None) -> DecomposedStrategy:
        strategy = next(self._collect_data_loop) if action is None else self._collect_data_loop.send(action)
        while not isinstance(strategy, DecomposedStrategy):
            strategy = next(self._collect_data_loop) if action is None else self._collect_data_loop.send(action)
        assert isinstance(strategy, DecomposedStrategy)
        return strategy

    def step(self, action: float) -> None:
        try:
            self._iter_strategy(action=action)
        except StopIteration:
            self._done = True

        self._maintainer.update(
            inner_executor=self._inner_executor,
            inner_strategy=self._inner_strategy,
        )

    def get_state(self) -> SAOEState:
        return SAOEState(
            order=self._order,
            cur_time=self._inner_executor.trade_calendar.get_step_time()[0],
            position=self._maintainer.position,
            history_exec=self._maintainer.history_exec,
            history_steps=self._maintainer.history_steps,
            metrics=self._maintainer.metrics,
            backtest_data=None,
            ticks_per_step=self._ticks_per_step,
            ticks_index=self._ticks_index,
            ticks_for_order=self._ticks_for_order,
        )

    def done(self) -> bool:
        return self._done

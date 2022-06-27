# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Placeholder for qlib-based simulator."""
import copy
from typing import Callable, Generator, List, Optional, Tuple, Union

import pandas as pd
from gym.vector.utils import spaces

from qlib.backtest import get_exchange
from qlib.backtest.account import Account
from qlib.backtest.decision import Order, TradeRange, TradeRangeByTime
from qlib.backtest.executor import BaseExecutor
from qlib.backtest.utils import CommonInfrastructure
from qlib.config import QlibConfig
from qlib.rl.interpreter import ActionInterpreter
from qlib.rl.order_execution.from_neutrader.config import ExchangeConfig
from qlib.rl.order_execution.from_neutrader.executor import RLNestedExecutor
from qlib.rl.order_execution.from_neutrader.feature import init_qlib
from qlib.rl.order_execution.from_neutrader.state import SAOEEpisodicState
from qlib.rl.order_execution.from_neutrader.strategy import DecomposedStrategy
from qlib.rl.simulator import Simulator
from qlib.strategy.base import BaseStrategy


def get_common_infra(
    config: ExchangeConfig,
    trade_start_time: pd.Timestamp,
    trade_end_time: pd.Timestamp,
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
        freq='1min',
        limit_threshold=config.limit_threshold,
        deal_price=config.deal_price,
        open_cost=config.open_cost,
        close_cost=config.close_cost,
        min_cost=config.min_cost if config.trade_unit is not None else 0,
        start_time=pd.Timestamp(trade_start_time),
        end_time=pd.Timestamp(trade_end_time) + pd.DateOffset(1),
        trade_unit=config.trade_unit,
        volume_threshold=config.volume_threshold
    )

    return CommonInfrastructure(trade_account=trade_account, trade_exchange=exchange)


class CategoricalActionInterpreter(ActionInterpreter[SAOEEpisodicState, int, float]):
    def __init__(self, values: Union[int, List[float]]) -> None:
        if isinstance(values, int):
            values = [i / values for i in range(0, values + 1)]
        self.action_values = values

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_values))

    def interpret(self, state: SAOEEpisodicState, action: int) -> float:
        volume = min(state.position, state.target * self.action_values[action])
        if state.cur_step + 1 >= state.num_step:
            volume = state.position  # execute all volumes at last
        return volume


class QlibSimulator(Simulator[Order, Tuple[SAOEEpisodicState, dict], float]):
    def __init__(
        self,
        time_per_step: str,
        start_time: str,
        end_time: str,
        qlib_config: QlibConfig,
        top_strategy_fn: Callable[[CommonInfrastructure, Order, TradeRange, str], BaseStrategy],
        inner_executor_fn: Callable[[CommonInfrastructure], BaseExecutor],
        exchange_config: ExchangeConfig,
    ) -> None:
        super(QlibSimulator, self).__init__(
            initial=None,  # TODO
        )

        self._trade_range = TradeRangeByTime(start_time, end_time)
        self._qlib_config = qlib_config
        self._time_per_step = time_per_step
        self._top_strategy_fn = top_strategy_fn
        self._inner_executor_fn = inner_executor_fn
        self._exchange_config = exchange_config

        self._executor: Optional[RLNestedExecutor] = None
        self._collect_data_loop: Optional[Generator] = None

        self._done = False

        self._inner_strategy = DecomposedStrategy()

    def reset(
        self,
        order: Order,
        instrument: str = "SH600000",  # TODO: Test only. Remove this default value later.
    ) -> None:
        init_qlib(self._qlib_config, instrument)

        common_infra = get_common_infra(
            self._exchange_config,
            trade_start_time=order.start_time,
            trade_end_time=order.end_time,
            codes=[instrument],
        )

        self._executor = RLNestedExecutor(
            time_per_step=self._time_per_step,
            inner_executor=self._inner_executor_fn(common_infra),
            inner_strategy=self._inner_strategy,
            track_data=True,
            common_infra=common_infra,
        )

        top_strategy = self._top_strategy_fn(common_infra, order, self._trade_range, instrument)

        self._executor.reset(start_time=order.start_time, end_time=order.end_time)
        top_strategy.reset(level_infra=self._executor.get_level_infra())

        self._collect_data_loop = self._executor.collect_data(top_strategy.generate_trade_decision(), level=0)
        assert isinstance(self._collect_data_loop, Generator)

        strategy = self._iter_strategy(action=None)
        sample, ep_state = strategy.sample_state_pair
        self._last_ep_state = ep_state
        self._last_info = self._collect_info(ep_state)

        self._done = False

    def _collect_info(self, ep_state: SAOEEpisodicState) -> dict:
        info = {
            "category": ep_state.flow_dir.value,
            # "reward": rew_info,  # TODO: ignore for now
        }
        if ep_state.done:
            # info["index"] = {"stock_id": sample.stock_id, "date": sample.date}  # TODO: ignore for now
            # info["history"] = {"action": self.action_history}  # TODO: ignore for now
            info.update(ep_state.logs())

            try:
                # done but loop is not exhausted
                # exhaust the loop manually
                while True:
                    self._collect_data_loop.send(0.)
            except StopIteration:
                pass

            info["qlib"] = {}
            for key, val in list(
                self._executor.trade_account.get_trade_indicator().order_indicator_his.values()
            )[0].to_series().items():
                info["qlib"][key] = val.item()

        return info

    def _iter_strategy(self, action: float = None) -> DecomposedStrategy:
        strategy = next(self._collect_data_loop) if action is None else self._collect_data_loop.send(action)
        while not isinstance(strategy, DecomposedStrategy):
            strategy = next(self._collect_data_loop) if action is None else self._collect_data_loop.send(action)
        assert isinstance(strategy, DecomposedStrategy)
        return strategy

    def step(self, action: float) -> None:
        try:
            strategy = self._iter_strategy(action=action)
            sample, ep_state = strategy.sample_state_pair
        except StopIteration:
            sample, ep_state = self._inner_strategy.sample_state_pair
            assert ep_state.done

        self._last_ep_state = ep_state
        self._last_info = self._collect_info(ep_state)

        if ep_state.done:
            self._done = True

    def get_state(self) -> Tuple[SAOEEpisodicState, dict]:
        return self._last_ep_state, self._last_info

    def done(self) -> bool:
        return self._done

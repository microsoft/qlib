# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import collections
from types import GeneratorType
from typing import Any, Callable, cast, Dict, Generator, List, Optional, Tuple, Union

import warnings
import numpy as np
import pandas as pd
import torch
from tianshou.data import Batch
from tianshou.policy import BasePolicy

from qlib.backtest import CommonInfrastructure, Order
from qlib.backtest.decision import BaseTradeDecision, TradeDecisionWithDetails, TradeDecisionWO, TradeRange
from qlib.backtest.exchange import Exchange
from qlib.backtest.executor import BaseExecutor
from qlib.backtest.utils import LevelInfrastructure, get_start_end_idx
from qlib.constant import EPS, ONE_MIN, REG_CN
from qlib.rl.data.native import IntradayBacktestData, load_backtest_data
from qlib.rl.interpreter import ActionInterpreter, StateInterpreter
from qlib.rl.order_execution.state import SAOEMetrics, SAOEState
from qlib.rl.order_execution.utils import dataframe_append, price_advantage
from qlib.strategy.base import RLStrategy
from qlib.utils import init_instance_by_config
from qlib.utils.index_data import IndexData
from qlib.utils.time import get_day_min_idx_range


def _get_all_timestamps(
    start: pd.Timestamp,
    end: pd.Timestamp,
    granularity: pd.Timedelta = ONE_MIN,
    include_end: bool = True,
) -> pd.DatetimeIndex:
    ret = []
    while start <= end:
        ret.append(start)
        start += granularity

    if ret[-1] > end:
        ret.pop()
    if ret[-1] == end and not include_end:
        ret.pop()
    return pd.DatetimeIndex(ret)


def fill_missing_data(
    original_data: np.ndarray,
    fill_method: Callable = np.nanmedian,
) -> np.ndarray:
    """Fill missing data.

    Parameters
    ----------
    original_data
        Original data without missing values.
    fill_method
        Method used to fill the missing data.

    Returns
    -------
        The filled data.
    """
    return np.nan_to_num(original_data, nan=fill_method(original_data))


class SAOEStateAdapter:
    """
    Maintain states of the environment. SAOEStateAdapter accepts execution results and update its internal state
    according to the execution results with additional information acquired from executors & exchange. For example,
    it gets the dealt order amount from execution results, and get the corresponding market price / volume from
    exchange.

    Example usage::

        adapter = SAOEStateAdapter(...)
        adapter.update(...)
        state = adapter.saoe_state
    """

    def __init__(
        self,
        order: Order,
        trade_decision: BaseTradeDecision,
        executor: BaseExecutor,
        exchange: Exchange,
        ticks_per_step: int,
        backtest_data: IntradayBacktestData,
        data_granularity: int = 1,
    ) -> None:
        self.position = order.amount
        self.order = order
        self.executor = executor
        self.exchange = exchange
        self.backtest_data = backtest_data
        self.start_idx, _ = get_start_end_idx(self.executor.trade_calendar, trade_decision)

        self.twap_price = self.backtest_data.get_deal_price().mean()

        metric_keys = list(SAOEMetrics.__annotations__.keys())  # pylint: disable=no-member
        self.history_exec = pd.DataFrame(columns=metric_keys).set_index("datetime")
        self.history_steps = pd.DataFrame(columns=metric_keys).set_index("datetime")
        self.metrics: Optional[SAOEMetrics] = None

        self.cur_time = max(backtest_data.ticks_for_order[0], order.start_time)
        self.ticks_per_step = ticks_per_step
        self.data_granularity = data_granularity
        assert self.ticks_per_step % self.data_granularity == 0

    def _next_time(self) -> pd.Timestamp:
        current_loc = self.backtest_data.ticks_index.get_loc(self.cur_time)
        next_loc = current_loc + (self.ticks_per_step // self.data_granularity)
        next_loc = next_loc - next_loc % (self.ticks_per_step // self.data_granularity)
        if (
            next_loc < len(self.backtest_data.ticks_index)
            and self.backtest_data.ticks_index[next_loc] < self.order.end_time
        ):
            return self.backtest_data.ticks_index[next_loc]
        else:
            return self.order.end_time

    def update(
        self,
        execute_result: list,
        last_step_range: Tuple[int, int],
    ) -> None:
        last_step_size = last_step_range[1] - last_step_range[0] + 1
        start_time = self.backtest_data.ticks_index[last_step_range[0]]
        end_time = self.backtest_data.ticks_index[last_step_range[1]]

        exec_vol = np.zeros(last_step_size)
        for order, _, __, ___ in execute_result:
            idx, _ = get_day_min_idx_range(order.start_time, order.end_time, f"{self.data_granularity}min", REG_CN)
            exec_vol[idx - last_step_range[0]] = order.deal_amount

        if exec_vol.sum() > self.position and exec_vol.sum() > 0.0:
            if exec_vol.sum() > self.position + 1.0:
                warnings.warn(
                    f"Sum of execution volume is {exec_vol.sum()} which is larger than "
                    f"position + 1.0 = {self.position} + 1.0 = {self.position + 1.0}. "
                    f"All execution volume is scaled down linearly to ensure that their sum does not position."
                )
            exec_vol *= self.position / (exec_vol.sum())

        market_volume = cast(
            IndexData,
            self.exchange.get_volume(
                self.order.stock_id,
                pd.Timestamp(start_time),
                pd.Timestamp(end_time),
                method=None,
            ),
        )
        market_price = cast(
            IndexData,
            self.exchange.get_deal_price(
                self.order.stock_id,
                pd.Timestamp(start_time),
                pd.Timestamp(end_time),
                method=None,
                direction=self.order.direction,
            ),
        )
        market_price = fill_missing_data(np.array(market_price, dtype=float).reshape(-1))
        market_volume = fill_missing_data(np.array(market_volume, dtype=float).reshape(-1))

        assert market_price.shape == market_volume.shape == exec_vol.shape

        # Get data from the current level executor's indicator
        current_trade_account = self.executor.trade_account
        current_df = current_trade_account.get_trade_indicator().generate_trade_indicators_dataframe()
        self.history_exec = dataframe_append(
            self.history_exec,
            self._collect_multi_order_metric(
                order=self.order,
                datetime=_get_all_timestamps(
                    start_time, end_time, include_end=True, granularity=ONE_MIN * self.data_granularity
                ),
                market_vol=market_volume,
                market_price=market_price,
                exec_vol=exec_vol,
                pa=current_df.iloc[-1]["pa"],
            ),
        )

        self.history_steps = dataframe_append(
            self.history_steps,
            [
                self._collect_single_order_metric(
                    self.order,
                    self.cur_time,
                    market_volume,
                    market_price,
                    exec_vol.sum(),
                    exec_vol,
                ),
            ],
        )

        # Do this at the end
        self.position -= exec_vol.sum()

        self.cur_time = self._next_time()

    def generate_metrics_after_done(self) -> None:
        """Generate metrics once the upper level execution is done"""

        self.metrics = self._collect_single_order_metric(
            self.order,
            self.backtest_data.ticks_index[0],  # start time
            self.history_exec["market_volume"],
            self.history_exec["market_price"],
            self.history_steps["amount"].sum(),
            self.history_exec["deal_amount"],
        )

    def _collect_multi_order_metric(
        self,
        order: Order,
        datetime: pd.DatetimeIndex,
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
            pa=price_advantage(exec_avg_price, self.twap_price, order.direction),
        )

    @property
    def saoe_state(self) -> SAOEState:
        return SAOEState(
            order=self.order,
            cur_time=self.cur_time,
            cur_step=self.executor.trade_calendar.get_trade_step() - self.start_idx,
            position=self.position,
            history_exec=self.history_exec,
            history_steps=self.history_steps,
            metrics=self.metrics,
            backtest_data=self.backtest_data,
            ticks_per_step=self.ticks_per_step,
            ticks_index=self.backtest_data.ticks_index,
            ticks_for_order=self.backtest_data.ticks_for_order,
        )


class SAOEStrategy(RLStrategy):
    """RL-based strategies that use SAOEState as state."""

    def __init__(
        self,
        policy: BasePolicy,
        outer_trade_decision: BaseTradeDecision | None = None,
        level_infra: LevelInfrastructure | None = None,
        common_infra: CommonInfrastructure | None = None,
        data_granularity: int = 1,
        **kwargs: Any,
    ) -> None:
        super(SAOEStrategy, self).__init__(
            policy=policy,
            outer_trade_decision=outer_trade_decision,
            level_infra=level_infra,
            common_infra=common_infra,
            **kwargs,
        )

        self._data_granularity = data_granularity
        self.adapter_dict: Dict[tuple, SAOEStateAdapter] = {}
        self._last_step_range = (0, 0)

    def _create_qlib_backtest_adapter(
        self,
        order: Order,
        trade_decision: BaseTradeDecision,
        trade_range: TradeRange,
    ) -> SAOEStateAdapter:
        backtest_data = load_backtest_data(order, self.trade_exchange, trade_range)

        return SAOEStateAdapter(
            order=order,
            trade_decision=trade_decision,
            executor=self.executor,
            exchange=self.trade_exchange,
            ticks_per_step=int(pd.Timedelta(self.trade_calendar.get_freq()) / ONE_MIN),
            backtest_data=backtest_data,
            data_granularity=self._data_granularity,
        )

    def reset(self, outer_trade_decision: BaseTradeDecision | None = None, **kwargs: Any) -> None:
        super(SAOEStrategy, self).reset(outer_trade_decision=outer_trade_decision, **kwargs)

        self.adapter_dict = {}
        self._last_step_range = (0, 0)

        if outer_trade_decision is not None and not outer_trade_decision.empty():
            trade_range = outer_trade_decision.trade_range
            assert trade_range is not None

            self.adapter_dict = {}
            for decision in outer_trade_decision.get_decision():
                order = cast(Order, decision)
                self.adapter_dict[order.key_by_day] = self._create_qlib_backtest_adapter(
                    order, outer_trade_decision, trade_range
                )

    def get_saoe_state_by_order(self, order: Order) -> SAOEState:
        return self.adapter_dict[order.key_by_day].saoe_state

    def post_upper_level_exe_step(self) -> None:
        for adapter in self.adapter_dict.values():
            adapter.generate_metrics_after_done()

    def post_exe_step(self, execute_result: Optional[list]) -> None:
        last_step_length = self._last_step_range[1] - self._last_step_range[0]
        if last_step_length <= 0:
            assert not execute_result
            return

        results = collections.defaultdict(list)
        if execute_result is not None:
            for e in execute_result:
                results[e[0].key_by_day].append(e)

        for key, adapter in self.adapter_dict.items():
            adapter.update(results[key], self._last_step_range)

    def generate_trade_decision(
        self,
        execute_result: list | None = None,
    ) -> Union[BaseTradeDecision, Generator[Any, Any, BaseTradeDecision]]:
        """
        For SAOEStrategy, we need to update the `self._last_step_range` every time a decision is generated.
        This operation should be invisible to developers, so we implement it in `generate_trade_decision()`
        The concrete logic to generate decisions should be implemented in `_generate_trade_decision()`.
        In other words, all subclass of `SAOEStrategy` should overwrite `_generate_trade_decision()` instead of
        `generate_trade_decision()`.
        """
        self._last_step_range = self.get_data_cal_avail_range(rtype="step")

        decision = self._generate_trade_decision(execute_result)
        if isinstance(decision, GeneratorType):
            decision = yield from decision

        return decision

    def _generate_trade_decision(
        self,
        execute_result: list | None = None,
    ) -> Union[BaseTradeDecision, Generator[Any, Any, BaseTradeDecision]]:
        raise NotImplementedError


class ProxySAOEStrategy(SAOEStrategy):
    """Proxy strategy that uses SAOEState. It is called a 'proxy' strategy because it does not make any decisions
    by itself. Instead, when the strategy is required to generate a decision, it will yield the environment's
    information and let the outside agents to make the decision. Please refer to `_generate_trade_decision` for
    more details.
    """

    def __init__(
        self,
        outer_trade_decision: BaseTradeDecision | None = None,
        level_infra: LevelInfrastructure | None = None,
        common_infra: CommonInfrastructure | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(None, outer_trade_decision, level_infra, common_infra, **kwargs)

    def _generate_trade_decision(self, execute_result: list | None = None) -> Generator[Any, Any, BaseTradeDecision]:
        # Once the following line is executed, this ProxySAOEStrategy (self) will be yielded to the outside
        # of the entire executor, and the execution will be suspended. When the execution is resumed by `send()`,
        # the item will be captured by `exec_vol`. The outside policy could communicate with the inner
        # level strategy through this way.
        exec_vol = yield self

        oh = self.trade_exchange.get_order_helper()
        order = oh.create(self._order.stock_id, exec_vol, self._order.direction)

        return TradeDecisionWO([order], self)

    def reset(self, outer_trade_decision: BaseTradeDecision | None = None, **kwargs: Any) -> None:
        super().reset(outer_trade_decision=outer_trade_decision, **kwargs)

        assert isinstance(outer_trade_decision, TradeDecisionWO)
        if outer_trade_decision is not None:
            order_list = outer_trade_decision.order_list
            assert len(order_list) == 1
            self._order = order_list[0]


class SAOEIntStrategy(SAOEStrategy):
    """(SAOE)state based strategy with (Int)preters."""

    def __init__(
        self,
        policy: dict | BasePolicy,
        state_interpreter: dict | StateInterpreter,
        action_interpreter: dict | ActionInterpreter,
        network: dict | torch.nn.Module | None = None,
        outer_trade_decision: BaseTradeDecision | None = None,
        level_infra: LevelInfrastructure | None = None,
        common_infra: CommonInfrastructure | None = None,
        **kwargs: Any,
    ) -> None:
        super(SAOEIntStrategy, self).__init__(
            policy=policy,
            outer_trade_decision=outer_trade_decision,
            level_infra=level_infra,
            common_infra=common_infra,
            **kwargs,
        )

        self._state_interpreter: StateInterpreter = init_instance_by_config(
            state_interpreter,
            accept_types=StateInterpreter,
        )
        self._action_interpreter: ActionInterpreter = init_instance_by_config(
            action_interpreter,
            accept_types=ActionInterpreter,
        )

        if isinstance(policy, dict):
            assert network is not None

            if isinstance(network, dict):
                network["kwargs"].update(
                    {
                        "obs_space": self._state_interpreter.observation_space,
                    }
                )
                network_inst = init_instance_by_config(network)
            else:
                network_inst = network

            policy["kwargs"].update(
                {
                    "obs_space": self._state_interpreter.observation_space,
                    "action_space": self._action_interpreter.action_space,
                    "network": network_inst,
                }
            )
            self._policy = init_instance_by_config(policy)
        elif isinstance(policy, BasePolicy):
            self._policy = policy
        else:
            raise ValueError(f"Unsupported policy type: {type(policy)}.")

        if self._policy is not None:
            self._policy.eval()

    def reset(self, outer_trade_decision: BaseTradeDecision | None = None, **kwargs: Any) -> None:
        super().reset(outer_trade_decision=outer_trade_decision, **kwargs)

    def _generate_trade_details(self, act: np.ndarray, exec_vols: List[float]) -> pd.DataFrame:
        assert hasattr(self.outer_trade_decision, "order_list")

        trade_details = []
        for a, v, o in zip(act, exec_vols, getattr(self.outer_trade_decision, "order_list")):
            trade_details.append(
                {
                    "instrument": o.stock_id,
                    "datetime": self.trade_calendar.get_step_time()[0],
                    "freq": self.trade_calendar.get_freq(),
                    "rl_exec_vol": v,
                }
            )
            if a is not None:
                trade_details[-1]["rl_action"] = a
        return pd.DataFrame.from_records(trade_details)

    def _generate_trade_decision(self, execute_result: list | None = None) -> BaseTradeDecision:
        states = []
        obs_batch = []
        for decision in self.outer_trade_decision.get_decision():
            order = cast(Order, decision)
            state = self.get_saoe_state_by_order(order)

            states.append(state)
            obs_batch.append({"obs": self._state_interpreter.interpret(state)})

        with torch.no_grad():
            policy_out = self._policy(Batch(obs_batch))
        act = policy_out.act.numpy() if torch.is_tensor(policy_out.act) else policy_out.act
        exec_vols = [self._action_interpreter.interpret(s, a) for s, a in zip(states, act)]

        oh = self.trade_exchange.get_order_helper()
        order_list = []
        for decision, exec_vol in zip(self.outer_trade_decision.get_decision(), exec_vols):
            if exec_vol != 0:
                order = cast(Order, decision)
                order_list.append(oh.create(order.stock_id, exec_vol, order.direction))

        return TradeDecisionWithDetails(
            order_list=order_list,
            strategy=self,
            details=self._generate_trade_details(act, exec_vols),
        )

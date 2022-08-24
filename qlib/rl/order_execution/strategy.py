# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import collections
from abc import ABCMeta
from types import GeneratorType
from typing import Any, cast, Dict, Generator, List, Optional, Union

import pandas as pd
import torch
from qlib.backtest import CommonInfrastructure, Exchange, Order
from qlib.backtest.decision import BaseTradeDecision, TradeDecisionWO, TradeRange
from qlib.backtest.utils import LevelInfrastructure
from qlib.constant import ONE_MIN
from qlib.rl.data.exchange_wrapper import load_qlib_backtest_data
from qlib.rl.interpreter import ActionInterpreter, StateInterpreter
from qlib.rl.order_execution.state import SAOEState, SAOEStateAdapter
from qlib.rl.utils import EnvWrapper
from qlib.rl.utils.env_wrapper import CollectDataEnvWrapper
from qlib.strategy.base import BaseStrategy, RLStrategy
from qlib.utils import init_instance_by_config
from tianshou.data import Batch
from tianshou.policy import BasePolicy


class SAOEStrategy(RLStrategy):
    """RL-based strategies that use SAOEState as state."""

    def __init__(
        self,
        policy: object,  # TODO: add accurate typehint later.
        outer_trade_decision: BaseTradeDecision = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
        **kwargs: Any,
    ) -> None:
        super(SAOEStrategy, self).__init__(
            policy=policy,
            outer_trade_decision=outer_trade_decision,
            level_infra=level_infra,
            common_infra=common_infra,
            **kwargs,
        )

        self.adapter_dict: Dict[tuple, SAOEStateAdapter] = {}
        self._last_step_range = (0, 0)

    def _create_qlib_backtest_adapter(self, order: Order, trade_range: TradeRange) -> SAOEStateAdapter:
        backtest_data = load_qlib_backtest_data(order, self.trade_exchange, trade_range)

        return SAOEStateAdapter(
            order=order,
            executor=self.executor,
            exchange=self.trade_exchange,
            ticks_per_step=int(pd.Timedelta(self.trade_calendar.get_freq()) / ONE_MIN),
            backtest_data=backtest_data,
        )

    def reset(self, outer_trade_decision: BaseTradeDecision = None, **kwargs: Any) -> None:
        super(SAOEStrategy, self).reset(outer_trade_decision=outer_trade_decision, **kwargs)

        self.adapter_dict = {}
        self._last_step_range = (0, 0)

        if outer_trade_decision is not None and not outer_trade_decision.empty():
            trade_range = outer_trade_decision.trade_range
            assert trade_range is not None

            self.adapter_dict = {}
            for decision in outer_trade_decision.get_decision():
                order = cast(Order, decision)
                self.adapter_dict[order.key_by_day] = self._create_qlib_backtest_adapter(order, trade_range)

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
        execute_result: list = None,
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
        execute_result: list = None,
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
        outer_trade_decision: BaseTradeDecision = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(None, outer_trade_decision, level_infra, common_infra, **kwargs)

    def _generate_trade_decision(self, execute_result: list = None) -> Generator[Any, Any, BaseTradeDecision]:
        # Once the following line is executed, this ProxySAOEStrategy (self) will be yielded to the outside
        # of the entire executor, and the execution will be suspended. When the execution is resumed by `send()`,
        # the item will be captured by `exec_vol`. The outside policy could communicate with the inner
        # level strategy through this way.
        exec_vol = yield self

        oh = self.trade_exchange.get_order_helper()
        order = oh.create(self._order.stock_id, exec_vol, self._order.direction)

        return TradeDecisionWO([order], self)

    def reset(self, outer_trade_decision: BaseTradeDecision = None, **kwargs: Any) -> None:
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
        network: object = None,  # TODO: add accurate typehint later.
        outer_trade_decision: BaseTradeDecision = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
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

    def set_env(self, env: EnvWrapper | CollectDataEnvWrapper) -> None:
        self._env = env
        self._state_interpreter.env = self._action_interpreter.env = self._env

    def reset(self, outer_trade_decision: BaseTradeDecision = None, **kwargs: Any) -> None:
        super().reset(outer_trade_decision=outer_trade_decision, **kwargs)

        if isinstance(self._env, CollectDataEnvWrapper):
            self._env.reset()

    def _generate_trade_decision(self, execute_result: list = None) -> BaseTradeDecision:
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

        if isinstance(self._env, CollectDataEnvWrapper):
            self._env.step(None)

        oh = self.trade_exchange.get_order_helper()
        order_list = []
        for decision, exec_vol in zip(self.outer_trade_decision.get_decision(), exec_vols):
            if exec_vol != 0:
                order = cast(Order, decision)
                order_list.append(oh.create(order.stock_id, exec_vol, order.direction))

        return TradeDecisionWO(order_list=order_list, strategy=self)


class MultiplexStrategyBase(BaseStrategy, metaclass=ABCMeta):
    def __init__(
        self,
        strategies: List[BaseStrategy] | List[dict],
        outer_trade_decision: BaseTradeDecision = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
        trade_exchange: Exchange = None,
    ) -> None:
        super().__init__(
            outer_trade_decision=outer_trade_decision,
            level_infra=level_infra,
            common_infra=common_infra,
            trade_exchange=trade_exchange,
        )

        self._strategies = [init_instance_by_config(strategy, accept_types=BaseStrategy) for strategy in strategies]

    def set_env(self, env: EnvWrapper | CollectDataEnvWrapper) -> None:
        for strategy in self._strategies:
            if hasattr(strategy, "set_env"):
                strategy.set_env(env)


class MultiplexStrategyOnTradeStep(MultiplexStrategyBase):
    """To use different strategy on different step of the outer calendar"""

    def __init__(
        self,
        strategies: List[BaseStrategy] | List[dict],
        outer_trade_decision: BaseTradeDecision = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
        trade_exchange: Exchange = None,
    ) -> None:
        super(MultiplexStrategyOnTradeStep, self).__init__(
            strategies=strategies,
            outer_trade_decision=outer_trade_decision,
            level_infra=level_infra,
            common_infra=common_infra,
            trade_exchange=trade_exchange,
        )

    def reset_level_infra(self, level_infra: LevelInfrastructure) -> None:
        for strategy in self._strategies:
            strategy.reset_level_infra(level_infra)

    def reset_common_infra(self, common_infra: CommonInfrastructure) -> None:
        for strategy in self._strategies:
            strategy.reset_common_infra(common_infra)

    def reset(self, outer_trade_decision: BaseTradeDecision = None, **kwargs: Any) -> None:
        super().reset(outer_trade_decision=outer_trade_decision, **kwargs)

        if outer_trade_decision is not None:
            strategy = self._get_current_strategy()
            strategy.reset(outer_trade_decision=outer_trade_decision, **kwargs)

    def generate_trade_decision(self, execute_result: list = None) -> BaseTradeDecision:
        if self.outer_trade_decision is not None:
            strategy = self._get_current_strategy()
            return strategy.generate_trade_decision(execute_result=execute_result)
        else:
            return TradeDecisionWO([], self)

    def post_exe_step(self, execute_result: list) -> None:
        if self.outer_trade_decision is not None:
            strategy = self._get_current_strategy()
            if isinstance(strategy, RLStrategy):
                strategy.post_exe_step(execute_result=execute_result)

    def _get_current_strategy(self) -> BaseStrategy:
        outer_calendar = self.outer_trade_decision.strategy.trade_calendar
        return self._strategies[outer_calendar.get_trade_step()]

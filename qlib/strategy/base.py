# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qlib.backtest.exchange import Exchange
    from qlib.backtest.position import BasePosition
from typing import List, Tuple, Union
import pandas as pd

from ..model.base import BaseModel
from ..data.dataset import DatasetH
from ..data.dataset.utils import convert_index_format
from ..rl.interpreter import ActionInterpreter, StateInterpreter
from ..utils import init_instance_by_config
from ..backtest.utils import CommonInfrastructure, LevelInfrastructure, TradeCalendarManager
from ..backtest.decision import BaseTradeDecision

__all__ = ["BaseStrategy", "RLStrategy", "RLIntStrategy"]


class BaseStrategy:
    """Base strategy for trading"""

    def __init__(
        self,
        outer_trade_decision: BaseTradeDecision = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
        trade_exchange: Exchange = None,
    ):
        """
        Parameters
        ----------
        outer_trade_decision : BaseTradeDecision, optional
            the trade decision of outer strategy which this strategy relies, and it will be traded in [start_time, end_time], by default None
            - If the strategy is used to split trade decision, it will be used
            - If the strategy is used for portfolio management, it can be ignored
        level_infra : LevelInfrastructure, optional
            level shared infrastructure for backtesting, including trade calendar
        common_infra : CommonInfrastructure, optional
            common infrastructure for backtesting, including trade_account, trade_exchange, .etc

        trade_exchange : Exchange
            exchange that provides market info, used to deal order and generate report
            - If `trade_exchange` is None, self.trade_exchange will be set with common_infra
            - It allowes different trade_exchanges is used in different executions.
            - For example:
                - In daily execution, both daily exchange and minutely are usable, but the daily exchange is recommended because it run faster.
                - In minutely execution, the daily exchange is not usable, only the minutely exchange is recommended.
        """

        self._reset(level_infra=level_infra, common_infra=common_infra, outer_trade_decision=outer_trade_decision)
        self._trade_exchange = trade_exchange

    @property
    def trade_calendar(self) -> TradeCalendarManager:
        return self.level_infra.get("trade_calendar")

    @property
    def trade_position(self) -> BasePosition:
        return self.common_infra.get("trade_account").current_position

    @property
    def trade_exchange(self) -> Exchange:
        """get trade exchange in a prioritized order"""
        return getattr(self, "_trade_exchange", None) or self.common_infra.get("trade_exchange")

    def reset_level_infra(self, level_infra: LevelInfrastructure):
        if not hasattr(self, "level_infra"):
            self.level_infra = level_infra
        else:
            self.level_infra.update(level_infra)

    def reset_common_infra(self, common_infra: CommonInfrastructure):
        if not hasattr(self, "common_infra"):
            self.common_infra: CommonInfrastructure = common_infra
        else:
            self.common_infra.update(common_infra)

    def reset(
        self,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
        outer_trade_decision=None,
        **kwargs,
    ):
        """
        - reset `level_infra`, used to reset trade calendar, .etc
        - reset `common_infra`, used to reset `trade_account`, `trade_exchange`, .etc
        - reset `outer_trade_decision`, used to make split decision

        **NOTE**:
        split this function into `reset` and `_reset` will make following cases more convenient
        1. Users want to initialize his strategy by overriding `reset`, but they don't want to affect the `_reset` called
        when initialization
        """
        self._reset(
            level_infra=level_infra, common_infra=common_infra, outer_trade_decision=outer_trade_decision, **kwargs
        )

    def _reset(
        self,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
        outer_trade_decision=None,
    ):
        """
        Please refer to the docs of `reset`
        """
        if level_infra is not None:
            self.reset_level_infra(level_infra)

        if common_infra is not None:
            self.reset_common_infra(common_infra)

        if outer_trade_decision is not None:
            self.outer_trade_decision = outer_trade_decision

    def generate_trade_decision(self, execute_result=None):
        """Generate trade decision in each trading bar

        Parameters
        ----------
        execute_result : List[object], optional
            the executed result for trade decision, by default None
            - When call the generate_trade_decision firstly, `execute_result` could be None
        """
        raise NotImplementedError("generate_trade_decision is not implemented!")

    def update_trade_decision(
        self, trade_decision: BaseTradeDecision, trade_calendar: TradeCalendarManager
    ) -> Union[BaseTradeDecision, None]:
        """
        update trade decision in each step of inner execution, this method enable all order

        Parameters
        ----------
        trade_decision : BaseTradeDecision
            the trade decision that will be updated
        trade_calendar : TradeCalendarManager
            The calendar of the **inner strategy**!!!!!

        Returns
        -------
            BaseTradeDecision:
        """
        # default to return None, which indicates that the trade decision is not changed
        return None

    def alter_outer_trade_decision(self, outer_trade_decision: BaseTradeDecision):
        """
        A method for updating the outer_trade_decision.
        The outer strategy may change its decision during updating.

        Parameters
        ----------
        outer_trade_decision : BaseTradeDecision
            the decision updated by the outer strategy
        """
        # default to reset the decision directly
        # NOTE: normally, user should do something to the strategy due to the change of outer decision
        raise NotImplementedError(f"Please implement the `alter_outer_trade_decision` method")

    # helper methods: not necessary but for convenience
    def get_data_cal_avail_range(self, rtype: str = "full") -> Tuple[int, int]:
        """
        return data calendar's available decision range for `self` strategy
        the range consider following factors
        - data calendar in the charge of `self` strategy
        - trading range limitation from the decision of outer strategy


        related methods
        - TradeCalendarManager.get_data_cal_range
        - BaseTradeDecision.get_data_cal_range_limit

        Parameters
        ----------
        rtype: str
            - "full": return the available data index range of the strategy from `start_time` to `end_time`
            - "step": return the available data index range of the strategy of current step

        Returns
        -------
        Tuple[int, int]:
            the available range both sides are closed
        """
        cal_range = self.trade_calendar.get_data_cal_range(rtype=rtype)
        if self.outer_trade_decision is None:
            raise ValueError(f"There is not limitation for strategy {self}")
        range_limit = self.outer_trade_decision.get_data_cal_range_limit(rtype=rtype)
        return max(cal_range[0], range_limit[0]), min(cal_range[1], range_limit[1])


class RLStrategy(BaseStrategy):
    """RL-based strategy"""

    def __init__(
        self,
        policy,
        outer_trade_decision: BaseTradeDecision = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        policy :
            RL policy for generate action
        """
        super(RLStrategy, self).__init__(outer_trade_decision, level_infra, common_infra, **kwargs)
        self.policy = policy


class RLIntStrategy(RLStrategy):
    """(RL)-based (Strategy) with (Int)erpreter"""

    def __init__(
        self,
        policy,
        state_interpreter: Union[dict, StateInterpreter],
        action_interpreter: Union[dict, ActionInterpreter],
        outer_trade_decision: BaseTradeDecision = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        state_interpreter : Union[dict, StateInterpreter]
            interpreter that interprets the qlib execute result into rl env state
        action_interpreter : Union[dict, ActionInterpreter]
            interpreter that interprets the rl agent action into qlib order list
        start_time : Union[str, pd.Timestamp], optional
            start time of trading, by default None
        end_time : Union[str, pd.Timestamp], optional
            end time of trading, by default None
        """
        super(RLIntStrategy, self).__init__(policy, outer_trade_decision, level_infra, common_infra, **kwargs)

        self.policy = policy
        self.state_interpreter = init_instance_by_config(state_interpreter, accept_types=StateInterpreter)
        self.action_interpreter = init_instance_by_config(action_interpreter, accept_types=ActionInterpreter)

    def generate_trade_decision(self, execute_result=None):
        _interpret_state = self.state_interpreter.interpret(execute_result=execute_result)
        _action = self.policy.step(_interpret_state)
        _trade_decision = self.action_interpreter.interpret(action=_action)
        return _trade_decision

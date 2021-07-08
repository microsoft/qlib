# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# TODO: rename it with decision.py
from __future__ import annotations
from enum import IntEnum
from qlib.log import get_module_logger

# try to fix circular imports when enabling type hints
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qlib.strategy.base import BaseStrategy
    from qlib.backtest.exchange import Exchange
from qlib.backtest.utils import TradeCalendarManager
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Union, List, Set, Tuple


class OrderDir(IntEnum):
    # Order  direction
    SELL = 0
    BUY = 1


@dataclass
class Order:
    """
    stock_id : str
    amount : float
    start_time : pd.Timestamp
        closed start time for order trading
    end_time : pd.Timestamp
        closed end time for order trading
    direction : int
        Order.SELL for sell; Order.BUY for buy
    factor : float
            presents the weight factor assigned in Exchange()
    """

    stock_id: str
    amount: float  # `amount` is a non-negative value

    # The interval of the order which belongs to (NOTE: this is not the expected order dealing range time)
    start_time: pd.Timestamp
    end_time: pd.Timestamp

    direction: int
    factor: float
    deal_amount: Optional[float] = None  # `deal_amount` is a non-negative value

    # FIXME:
    # for compatible now.
    # Plese remove them in the future
    SELL: ClassVar[OrderDir] = OrderDir.SELL
    BUY: ClassVar[OrderDir] = OrderDir.BUY

    def __post_init__(self):
        if self.direction not in {Order.SELL, Order.BUY}:
            raise NotImplementedError("direction not supported, `Order.SELL` for sell, `Order.BUY` for buy")
        self.deal_amount = 0

    @property
    def amount_delta(self) -> float:
        """
        return the delta of amount.
        - Positive value indicates buying `amount` of share
        - Negative value indicates selling `amount` of share
        """
        return self.amount * self.sign

    @property
    def deal_amount_delta(self) -> float:
        """
        return the delta of deal_amount.
        - Positive value indicates buying `deal_amount` of share
        - Negative value indicates selling `deal_amount` of share
        """
        return self.deal_amount * self.sign

    @property
    def sign(self) -> float:
        """
        return the sign of trading
        - `+1` indicates buying
        - `-1` value indicates selling
        """
        return self.direction * 2 - 1

    @staticmethod
    def parse_dir(direction: Union[str, int, np.integer, OrderDir]) -> OrderDir:
        if isinstance(direction, OrderDir):
            return direction
        elif isinstance(direction, (int, float, np.integer, np.floating)):
            if direction > 0:
                return Order.BUY
            else:
                return Order.SELL
        elif isinstance(direction, str):
            dl = direction.lower()
            if dl.strip() == "sell":
                return OrderDir.SELL
            elif dl.strip() == "buy":
                return OrderDir.BUY
            else:
                raise NotImplementedError(f"This type of input is not supported")
        else:
            raise NotImplementedError(f"This type of input is not supported")


class OrderHelper:
    """
    Motivation
    - Make generating order easier
        - User may have no knowledge about the adjust-factor information about the system.
        - It involves to much interaction with the exchange when generating orders.
    """

    def __init__(self, exchange: Exchange):
        self.exchange = exchange

    def create(
        self,
        code: str,
        amount: float,
        direction: OrderDir,
        start_time: Union[str, pd.Timestamp],
        end_time: Union[str, pd.Timestamp],
    ) -> Order:
        """
        help to create a order

        # TODO: create order for unadjusted amount order

        Parameters
        ----------
        code : str
            the id of the instrument
        amount : float
            **adjusted trading amount**
        direction : OrderDir
            trading  direction
        start_time : Union[str, pd.Timestamp]
            The interval of the order which belongs to
        end_time : Union[str, pd.Timestamp]
            The interval of the order which belongs to

        Returns
        -------
        Order:
            The created order
        """
        start_time = pd.Timestamp(start_time)
        end_time = pd.Timestamp(end_time)
        return Order(
            stock_id=code,
            amount=amount,
            start_time=start_time,
            end_time=end_time,
            direction=direction,
            factor=self.exchange.get_factor(code, start_time, end_time),
        )


class BaseTradeDecision:
    """
    Trade decisions ara made by strategy and executed by exeuter

    Motivation:
        Here are several typical scenarios for `BaseTradeDecision`

        Case 1:
        1. Outer strategy makes a decision. The decision is not available at the start of current interval
        2. After a period of time, the decision are updated and become available
        3. The inner strategy try to get the decision and start to execute the decision according to `get_range_limit`
        Case 2:
        1. The outer strategy's decision is available at the start of the interval
        2. Same as `case 1.3`
    """

    def __init__(self, strategy: BaseStrategy, idx_range: Tuple[int, int] = None):
        """
        Parameters
        ----------
        strategy : BaseStrategy
            The strategy who make the decision
        """
        self.strategy = strategy
        self.total_step = None  # upper strategy has no knowledge about the sub executor before `_init_sub_trading`
        self.idx_range = idx_range

    def get_decision(self) -> List[object]:
        """
        get the **concrete decision**  (e.g. execution orders)
        This will be called by the inner strategy

        Returns
        -------
        List[object]:
            The decision result. Typically it is some orders
            Example:
                []:
                    Decision not available
                [concrete_decision]:
                    available
        """
        raise NotImplementedError(f"This type of input is not supported")

    def update(self, trade_calendar: TradeCalendarManager) -> Union["BaseTradeDecision", None]:
        """
        Be called at the **start** of each step.

        This function is designn for following purpose
        1) Leave a hook for the strategy who make `self` decision to update the decision itself
        2) Update some information from the inner executor calendar

        Parameters
        ----------
        trade_calendar : TradeCalendarManager
            The calendar of the **inner strategy**!!!!!

        Returns
        -------
        None:
            No update, use previous decision(or unavailable)
        BaseTradeDecision:
            New update, use new decision
        """
        # purpose 1)
        self.total_step = trade_calendar.get_trade_len()
        if self.idx_range is not None:
            logger = get_module_logger("decision")
            start_idx, end_idx = self.idx_range
            if start_idx < 0 or end_idx >= self.total_step:
                logger.warning(f"{self.idx_range} go beyound the total_step({self.total_step}), it will be clipped")
                self.idx_range = max(0, start_idx), min(self.total_step - 1, end_idx)

        # purpose 2)
        return self.strategy.update_trade_decision(self, trade_calendar)

    def get_range_limit(self, **kwargs) -> Tuple[int, int]:
        """
        return the expected step range for limiting the decision execution time
        Both left and right are **closed**

        **kwargs:
            {"default_value": <default_value>}
            # using dict is for distinguish no value provided or None provided

        Returns
        -------
        Tuple[int, int]:

        Raises
        ------
        NotImplementedError:
            If the following criteria meet
            1) the decision can't provide a unified start and end
            2) default_value is None
        """
        if self.idx_range is None:
            if "default_value" in kwargs:
                return kwargs["default_value"]
            else:
                # Default to get full index
                raise NotImplementedError(f"The decision didn't provide an index range")
        return self.idx_range

    def empty(self) -> bool:
        for obj in self.get_decision():
            if isinstance(obj, Order):
                # Zero amount order will be treated as empty
                if not np.isclose(obj.amount, 0.0):
                    return False
            else:
                return True
        return True


class EmptyTradeDecision(BaseTradeDecision):
    def empty(self) -> bool:
        return True


class TradeDecisionWO(BaseTradeDecision):
    """
    Trade Decision (W)ith (O)rder.
    Besides, the time_range is also included.
    """

    def __init__(self, order_list: List[Order], strategy: BaseStrategy, idx_range: Tuple[int, int] = None):
        super().__init__(strategy, idx_range=idx_range)
        self.order_list = order_list

    def get_decision(self) -> List[object]:
        return self.order_list

    def __repr__(self) -> str:
        return f"strategy: {self.strategy}; idx_range: {self.idx_range}; order_list[{len(self.order_list)}]"


# TODO: the orders below need to be discussed ------------------------------------
# - The classes below are designed for Case 1
# - However, Case 1 can't take `order_pool` as the an argument as the constructor function
class TradeDecisionWithOrderPool:
    """trade decision that made by strategy"""

    def __init__(self, strategy, order_pool):
        """
        Parameters
        ----------
        strategy : BaseStrategy
            the original strategy that make the decision
        order_pool : list, optional
            the candinate order pool for generate trade decision
        """
        super(TradeDecisionWithOrderPool, self).__init__(strategy)
        self.order_pool = order_pool
        self.order_list = []

    def pop_order_pool(self, pop_len):
        if pop_len > len(self.order_pool):
            warnings.warn(
                f"pop len {pop_len} is too much length than order pool, cut it as pool length {len(self.order_pool)}"
            )
            pop_len = len(self.order_pool)
        res = self.order_pool[:pop_len]
        del self.order_pool[:pop_len]
        return res

    def push_order_list(self, order_list):
        self.order_list.extend(order_list)

    def get_decision(self):
        """get the order list

        Parameters
        ----------
        only_enable : bool, optional
            wether to ignore disabled order, by default False
        only_disable : bool, optional
            wether to ignore enabled order, by default False
        Returns
        -------
        List[Order]
            the order list
        """
        return self.order_list

    def update(self, trade_calendar):
        """make the original strategy update the enabled status of orders."""
        self.ori_strategy.update_trade_decision(self, trade_calendar)


class BaseDecisionUpdater:
    def update_decision(self, decision, trade_calendar) -> BaseTradeDecision:
        """
        Parameters
        ----------
        decision : BaseTradeDecision
            the trade decision to be updated
        trade_calendar : BaseTradeCalendar
            the trade calendar of inner execution

        Returns
        -------
        BaseTradeDecision
            the updated decision
        """
        raise NotImplementedError(f"This method is not implemented")


class DecisionUpdaterWithOrderPool:
    def __init__(self, plan_config=None):
        """
        Parameters
        ----------
        plan_config : Dict[Tuple(int, float)], optional
            the plan config, by default None
        """
        if plan_config is None:
            self.plan_config = [(0, 1)]
        else:
            self.plan_config = plan_config

    def update_decision(self, decision, trade_calendar) -> BaseTradeDecision:
        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        for _index, _ratio in self.plan_config:
            if trade_step == _index:
                pop_len = len(decision.order_pool) * _ratio
                pop_order_list = decision.pop_order_pool(pop_len)
                decision.push_order_list(pop_order_list)

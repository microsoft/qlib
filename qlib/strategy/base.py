# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
from typing import List, Union


from ..model.base import BaseModel
from ..data.dataset import DatasetH
from ..data.dataset.utils import convert_index_format
from ..contrib.backtest.order import Order
from ..contrib.backtest.executor import BaseTradeCalendar
from ..rl.interpreter import ActionInterpreter, StateInterpreter


class BaseStrategy(BaseTradeCalendar):
    """Base strategy for trading"""

    def generate_order_list(self, execute_state):
        """Generate order list in each trading bar"""
        raise NotImplementedError("generator_order_list is not implemented!")


class RuleStrategy(BaseStrategy):
    """Rule-based Trading strategy"""

    pass


class ModelStrategy(BaseStrategy):
    """Model-based trading strategy, use model to make predictions for trading"""

    def __init__(
        self,
        step_bar: str,
        model: BaseModel,
        dataset: DatasetH,
        start_time: Union[str, pd.Timestamp] = None,
        end_time: Union[str, pd.Timestamp] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        model : BaseModel
            the model used in when making predictions
        dataset : DatasetH
            provide test data for model
        kwargs : dict
            arguments that will be passed into `reset` method
        """
        self.model = model
        self.dataset = dataset
        self.pred_scores = convert_index_format(self.model.predict(dataset), level="datetime")
        # pred_score_dates = self.pred_scores.index.get_level_values(level="datetime")
        super(ModelStrategy, self).__init__(step_bar, start_time, end_time, **kwargs)

    def _update_model(self):
        """
        When using online data, pdate model in each bar as the following steps:
            - update dataset with online data, the dataset should support online update
            - make the latest prediction scores of the new bar
            - update the pred score into the latest prediction
        """
        raise NotImplementedError("_update_model is not implemented!")


class RLStrategy(BaseStrategy):
    """RL-based strategy"""

    def __init__(
        self,
        step_bar: str,
        policy,
        start_time: Union[str, pd.Timestamp] = None,
        end_time: Union[str, pd.Timestamp] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        policy :
            RL policy for generate action
        """
        super(RLStrategy, self).__init__(step_bar, start_time, end_time, **kwargs)
        self.policy = policy


class RLIntStrategy(RLStrategy):
    """(RL)-based (Strategy) with (Int)erpreter"""

    def __init__(
        self,
        step_bar: str,
        policy,
        state_interpreter: StateInterpreter,
        action_interpreter: ActionInterpreter,
        start_time: Union[str, pd.Timestamp] = None,
        end_time: Union[str, pd.Timestamp] = None,
        state_interpret_kwargs: dict = {},
        action_interpret_kwargs: dict = {},
        **kwargs,
    ):
        """
        Parameters
        ----------
        state_interpreter : StateInterpreter
            interpretor that interprets the qlib execute result into rl env state.
        action_interpreter : ActionInterpreter
            interpretor that interprets the rl agent action into qlib order list
        start_time : Union[str, pd.Timestamp], optional
            start time of trading, by default None
        end_time : Union[str, pd.Timestamp], optional
            end time of trading, by default None
        state_interpret_kwargs : dict, optional
            arguments may be used in `state_interpreter.interpret`, by default {}
            such as the following arguments:
                - trade exchange : Exchange
                    Exchange that can provide market info
        action_interpret_kwargs: dict, optional
            arguments may be used in `action_interpreter.interpret`, by default {}
            such as the following arguments:
                - trade_order_list : List[Order]
                    If the strategy is used to split order, it presents the trade order pool.
        """
        super(RLIntStrategy, self).__init__(step_bar, policy, start_time, end_time, **kwargs)

        self.policy = policy
        self.action_interpreter = action_interpreter
        self.state_interpreter = state_interpreter
        self.state_interpret_kwargs = state_interpret_kwargs
        self.action_interpret_kwargs = action_interpret_kwargs

    def generate_order_list(self, execute_state):
        super(RLStrategy, self).step()
        _interpret_state = self.state_interpretor.interpret(
            execute_result=execute_state, **self.action_interpret_kwargs
        )
        _policy_action = self.policy.step(_interpret_state)
        _order_list = self.action_interpreter.interpret(action=_policy_action, **self.state_interpret_kwargs)
        return _order_list


class OrderEnhancement:
    """
    Order enhancement for strategy
        - If the strategy is used to split orders, the enhancement should be inherited
        - If the strategy is used for portfolio management, the enhancement can be ignored
    """

    def reset(self, trade_order_list: List[Order] = None):
        """reset trade orders for split strategy

        Parameters
        ----------
        trade_order_list for split strategy: List[Order], optional
            trading orders , by default None
        """
        if trade_order_list is not None:
            self.trade_order_list = trade_order_list

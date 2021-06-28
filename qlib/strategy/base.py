# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import List, Union

from ..model.base import BaseModel
from ..data.dataset import DatasetH
from ..data.dataset.utils import convert_index_format
from ..rl.interpreter import ActionInterpreter, StateInterpreter
from ..utils import init_instance_by_config
from ..backtest.utils import CommonInfrastructure, LevelInfrastructure, TradeCalendarManager
from ..backtest.order import BaseTradeDecision


class BaseStrategy:
    """Base strategy for trading"""

    def __init__(
        self,
        outer_trade_decision: BaseTradeDecision = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
    ):
        """
        Parameters
        ----------
        outer_trade_decision : BaseTradeDecision, optional
            the trade decision of outer strategy which this startegy relies, and it will be traded in [start_time, end_time], by default None
            - If the strategy is used to split trade decision, it will be used
            - If the strategy is used for portfolio management, it can be ignored
        level_infra : LevelInfrastructure, optional
            level shared infrastructure for backtesting, including trade calendar
        common_infra : CommonInfrastructure, optional
            common infrastructure for backtesting, including trade_account, trade_exchange, .etc
        """

        self.reset(level_infra=level_infra, common_infra=common_infra, outer_trade_decision=outer_trade_decision)

    def reset_level_infra(self, level_infra: LevelInfrastructure):
        if not hasattr(self, "level_infra"):
            self.level_infra = level_infra
        else:
            self.level_infra.update(level_infra)

        if level_infra.has("trade_calendar"):
            self.trade_calendar: TradeCalendarManager = level_infra.get("trade_calendar")

    def reset_common_infra(self, common_infra: CommonInfrastructure):
        if not hasattr(self, "common_infra"):
            self.common_infra: CommonInfrastructure = common_infra
        else:
            self.common_infra.update(common_infra)

        if common_infra.has("trade_account"):
            self.trade_position = common_infra.get("trade_account").current

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


class ModelStrategy(BaseStrategy):
    """Model-based trading strategy, use model to make predictions for trading"""

    def __init__(
        self,
        model: BaseModel,
        dataset: DatasetH,
        outer_trade_decision: BaseTradeDecision = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
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
        super(ModelStrategy, self).__init__(outer_trade_decision, level_infra, common_infra, **kwargs)
        self.model = model
        self.dataset = dataset
        self.pred_scores = convert_index_format(self.model.predict(dataset), level="datetime")

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
            interpretor that interprets the qlib execute result into rl env state
        action_interpreter : Union[dict, ActionInterpreter]
            interpretor that interprets the rl agent action into qlib order list
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

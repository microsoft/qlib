# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
This strategy is not well maintained
"""


from .order_generator import OrderGenWInteract
from .signal_strategy import WeightStrategyBase
import copy


class SoftTopkStrategy(WeightStrategyBase):
    def __init__(
        self,
        model,
        dataset,
        topk,
        order_generator_cls_or_obj=OrderGenWInteract,
        max_sold_weight=1.0,
        risk_degree=0.95,
        buy_method="first_fill",
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
        **kwargs,
    ):
        """Parameter
        topk : int
            top-N stocks to buy
        risk_degree : float
            position percentage of total value
            buy_method :
                rank_fill: assign the weight stocks that rank high first(1/topk max)
                average_fill: assign the weight to the stocks rank high averagely.
        """
        super(SoftTopkStrategy, self).__init__(
            model, dataset, order_generator_cls_or_obj, trade_exchange, level_infra, common_infra, **kwargs
        )
        self.topk = topk
        self.max_sold_weight = max_sold_weight
        self.risk_degree = risk_degree
        self.buy_method = buy_method

    def get_risk_degree(self, trade_step=None):
        """get_risk_degree
        Return the proportion of your total value you will used in investment.
        Dynamically risk_degree will result in Market timing
        """
        # It will use 95% amount of your total value by default
        return self.risk_degree

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        """Parameter:
        score : pred score for this trade date, pd.Series, index is stock_id, contain 'score' column
        current : current position, use Position() class
        trade_date : trade date
        generate target position from score for this date and the current position
        The cache is not considered in the position
        """
        # TODO:
        # If the current stock list is more than topk(eg. The weights are modified
        # by risk control), the weight will not be handled correctly.
        buy_signal_stocks = set(score.sort_values(ascending=False).iloc[: self.topk].index)
        cur_stock_weight = current.get_stock_weight_dict(only_stock=True)

        if len(cur_stock_weight) == 0:
            final_stock_weight = {code: 1 / self.topk for code in buy_signal_stocks}
        else:
            final_stock_weight = copy.deepcopy(cur_stock_weight)
            sold_stock_weight = 0.0
            for stock_id in final_stock_weight:
                if stock_id not in buy_signal_stocks:
                    sw = min(self.max_sold_weight, final_stock_weight[stock_id])
                    sold_stock_weight += sw
                    final_stock_weight[stock_id] -= sw
            if self.buy_method == "first_fill":
                for stock_id in buy_signal_stocks:
                    add_weight = min(
                        max(1 / self.topk - final_stock_weight.get(stock_id, 0), 0.0),
                        sold_stock_weight,
                    )
                    final_stock_weight[stock_id] = final_stock_weight.get(stock_id, 0.0) + add_weight
                    sold_stock_weight -= add_weight
            elif self.buy_method == "average_fill":
                for stock_id in buy_signal_stocks:
                    final_stock_weight[stock_id] = final_stock_weight.get(stock_id, 0.0) + sold_stock_weight / len(
                        buy_signal_stocks
                    )
            else:
                raise ValueError("Buy method not found")
        return final_stock_weight

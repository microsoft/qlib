# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
智能组合优化器

基于风险约束的投资组合优化，生成目标权重和交易订单，包括：
- 凸优化求解最优权重
- 风险模型集成
- 订单生成
- 风险分析和归因
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import qlib
from qlib.contrib.strategy.optimizer.enhanced_indexing import EnhancedIndexingOptimizer
from loguru import logger


class PortfolioOptimizer:
    """智能组合优化器

    基于信号和当前持仓生成优化的目标权重和交易订单

    Attributes:
        market (str): 市场代码
        optimizer: 优化器实例

    Examples:
        >>> optimizer = PortfolioOptimizer(market='cn')
        >>> plan = optimizer.generate_rebalance_plan(
        ...     signals=signals,
        ...     current_positions={'SH600000': 0.05},
        ...     benchmark_weights={'SH600000': 0.03}
        ... )
        >>> print(plan['orders'])
    """

    def __init__(
        self,
        market: str = 'cn',
        lamb: float = 1.0,
        max_turnover: float = 0.3,
        max_position_deviation: float = 0.05,
        factor_deviation: Optional[np.ndarray] = None,
    ):
        """初始化组合优化器

        Args:
            market: 市场代码
            lamb: 风险厌恶系数 (越大越保守)
            max_turnover: 最大换手率
            max_position_deviation: 单股票最大偏离基准权重
            factor_deviation: 因子偏离限制
        """
        self.market = market

        # 初始化优化器
        self.optimizer = EnhancedIndexingOptimizer(
            lamb=lamb,
            delta=max_turnover,
            b_dev=max_position_deviation,
            f_dev=factor_deviation,
            scale_return=True,
            epsilon=5e-5,
        )

        # 配置参数
        self.config = {
            'lamb': lamb,
            'max_turnover': max_turnover,
            'max_position_deviation': max_position_deviation,
            'min_weight': 5e-5,  # 最小权重
            'max_weight': 0.1,   # 单只股票最大权重
        }

        logger.info(
            f"组合优化器初始化 [风险厌恶: {lamb}, "
            f"最大换手: {max_turnover:.1%}, 最大偏离: {max_position_deviation:.1%}]"
        )

    def generate_rebalance_plan(
        self,
        signals: pd.Series,
        current_positions: Dict[str, float],
        benchmark_weights: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict] = None,
        total_value: float = 100000000,  # 1亿资金
    ) -> Dict:
        """生成再平衡计划

        Args:
            signals: 预测信号 (股票代码 -> 分数)
            current_positions: 当前持仓 (股票代码 -> 权重)
            benchmark_weights: 基准权重 (股票代码 -> 权重)，None表示等权
            constraints: 额外约束 {'must_hold': [...], 'must_sell': [...]}
            total_value: 总资产价值

        Returns:
            Dict: 再平衡计划
                - target_weights: 目标权重
                - orders: 订单列表
                - risk_analysis: 风险分析
                - turnover: 换手率

        Examples:
            >>> plan = optimizer.generate_rebalance_plan(
            ...     signals=pd.Series({'SH600000': 0.8, 'SH600036': 0.6}),
            ...     current_positions={'SH600000': 0.05},
            ...     benchmark_weights=None
            ... )
        """
        logger.info("开始生成组合优化方案...")

        # 1. 准备数据
        stocks = list(signals.index)
        n = len(stocks)

        if n == 0:
            raise ValueError("信号为空，无法优化")

        logger.info(f"优化股票池: {n} 只股票")

        # 预期收益 (来自模型信号)
        r = signals.reindex(stocks, fill_value=0).values

        # 当前权重
        w0 = np.array([current_positions.get(s, 0) for s in stocks])

        # 基准权重 (如果未提供，使用等权)
        if benchmark_weights is None:
            wb = np.ones(n) / n
        else:
            wb = np.array([benchmark_weights.get(s, 1/n) for s in stocks])

        # 归一化权重 (确保和为1)
        w0_sum = w0.sum()
        wb_sum = wb.sum()

        if w0_sum > 0:
            w0 = w0 / w0_sum
        if wb_sum > 0:
            wb = wb / wb_sum

        # 2. 获取因子暴露和风险模型
        risk_data = self._build_risk_model(stocks)
        F = risk_data['factor_exposure']   # (n_stocks, n_factors)
        cov_b = risk_data['factor_cov']     # (n_factors, n_factors)
        var_u = risk_data['specific_var']   # (n_stocks,)

        # 3. 处理约束
        mfh = None  # 强制持有标记
        mfs = None  # 强制卖出标记

        if constraints:
            if 'must_hold' in constraints:
                must_hold_indices = [i for i, s in enumerate(stocks) if s in constraints['must_hold']]
                if must_hold_indices:
                    mfh = np.zeros(n, dtype=bool)
                    mfh[must_hold_indices] = True

            if 'must_sell' in constraints:
                must_sell_indices = [i for i, s in enumerate(stocks) if s in constraints['must_sell']]
                if must_sell_indices:
                    mfs = np.zeros(n, dtype=bool)
                    mfs[must_sell_indices] = True

        # 4. 运行优化
        try:
            logger.info("运行凸优化求解...")
            w_target = self.optimizer(
                r=r,
                F=F,
                cov_b=cov_b,
                var_u=var_u,
                w0=w0,
                wb=wb,
                mfh=mfh,
                mfs=mfs,
            )

            # 清理极小权重
            w_target[w_target < self.config['min_weight']] = 0

            # 重新归一化
            w_target = w_target / w_target.sum()

        except Exception as e:
            logger.error(f"优化失败: {str(e)}")
            # 降级方案：使用信号排名生成等权组合
            logger.warning("使用降级方案：Top-K等权组合")
            w_target = self._fallback_topk_portfolio(signals, topk=30)
            w_target = w_target.reindex(stocks, fill_value=0).values

        # 5. 生成交易订单
        orders_df = self._generate_orders(
            stocks=stocks,
            w_current=w0,
            w_target=w_target,
            current_positions=current_positions,
            total_value=total_value
        )

        # 6. 风险分析
        risk_analysis = self._analyze_risk(
            w_target=w_target,
            F=F,
            cov_b=cov_b,
            var_u=var_u,
            wb=wb,
            stocks=stocks
        )

        # 7. 计算换手率
        turnover = np.sum(np.abs(w_target - w0))

        result = {
            'target_weights': pd.Series(w_target, index=stocks),
            'orders': orders_df,
            'risk_analysis': risk_analysis,
            'turnover': turnover,
            'n_buy': int((orders_df['direction'] == 'BUY').sum()) if len(orders_df) > 0 else 0,
            'n_sell': int((orders_df['direction'] == 'SELL').sum()) if len(orders_df) > 0 else 0,
            'n_hold': n - len(orders_df),
        }

        logger.success(
            f"✓ 优化完成 [换手率: {turnover:.2%}, "
            f"买入: {result['n_buy']}, 卖出: {result['n_sell']}, "
            f"跟踪误差: {risk_analysis['tracking_error']:.2%}]"
        )

        return result

    def _build_risk_model(self, stocks: List[str]) -> Dict:
        """构建风险模型

        Args:
            stocks: 股票列表

        Returns:
            Dict: 风险模型数据
        """
        n = len(stocks)

        # 简化的风险模型
        # 实际应用中应该使用 Barra CNE6 等专业风险模型

        # 生成10个因子的暴露
        n_factors = 10

        # 模拟因子暴露 (可以从基本面、技术面、风格因子等获取)
        factor_exposure = np.random.randn(n, n_factors) * 0.1

        # 因子协方差矩阵 (应该从历史数据估计)
        factor_cov = np.eye(n_factors) * 0.01

        # 特异性风险 (个股特有风险)
        specific_var = np.ones(n) * 0.02

        return {
            'factor_exposure': factor_exposure,
            'factor_cov': factor_cov,
            'specific_var': specific_var,
        }

    def _generate_orders(
        self,
        stocks: List[str],
        w_current: np.ndarray,
        w_target: np.ndarray,
        current_positions: Dict[str, float],
        total_value: float
    ) -> pd.DataFrame:
        """生成交易订单列表

        Args:
            stocks: 股票列表
            w_current: 当前权重
            w_target: 目标权重
            current_positions: 当前持仓字典
            total_value: 总资产价值

        Returns:
            pd.DataFrame: 订单列表
        """
        orders = []

        for i, stock in enumerate(stocks):
            delta_w = w_target[i] - w_current[i]

            # 忽略微小变化
            if abs(delta_w) < 1e-4:
                continue

            order = {
                'stock': stock,
                'direction': 'BUY' if delta_w > 0 else 'SELL',
                'current_weight': w_current[i],
                'target_weight': w_target[i],
                'delta_weight': delta_w,
                'amount_value': abs(delta_w) * total_value,  # 交易金额
                'amount_shares': None,  # 股数 (需要价格信息)
            }

            orders.append(order)

        # 按交易金额排序
        orders_df = pd.DataFrame(orders)

        if len(orders_df) > 0:
            orders_df = orders_df.sort_values('amount_value', ascending=False)

        return orders_df

    def _analyze_risk(
        self,
        w_target: np.ndarray,
        F: np.ndarray,
        cov_b: np.ndarray,
        var_u: np.ndarray,
        wb: np.ndarray,
        stocks: List[str]
    ) -> Dict:
        """风险分析

        Args:
            w_target: 目标权重
            F: 因子暴露矩阵
            cov_b: 因子协方差矩阵
            var_u: 特异性风险
            wb: 基准权重
            stocks: 股票列表

        Returns:
            Dict: 风险分析结果
        """
        d = w_target - wb  # 相对基准的偏离
        v = d @ F          # 因子暴露偏离

        # 计算跟踪误差 (年化)
        tracking_variance = v @ cov_b @ v + var_u @ (d**2)
        tracking_error = np.sqrt(tracking_variance * 252)

        # 因子风险贡献
        factor_risk_contrib = v @ cov_b @ v

        # 特异性风险贡献
        specific_risk_contrib = var_u @ (d**2)

        # 持仓集中度 (Herfindahl指数)
        concentration = (w_target**2).sum()

        # Top10持仓权重
        top10_weight = np.sort(w_target)[-10:].sum()

        # 有效股票数 (1 / HHI)
        effective_n = 1 / concentration if concentration > 0 else 0

        return {
            'tracking_error': tracking_error,
            'factor_risk': factor_risk_contrib,
            'specific_risk': specific_risk_contrib,
            'risk_decomp': {
                'factor': factor_risk_contrib / (factor_risk_contrib + specific_risk_contrib + 1e-8),
                'specific': specific_risk_contrib / (factor_risk_contrib + specific_risk_contrib + 1e-8),
            },
            'max_position': w_target.max(),
            'min_position': w_target[w_target > 0].min() if (w_target > 0).any() else 0,
            'concentration': concentration,
            'top10_weight': top10_weight,
            'effective_n_stocks': effective_n,
            'n_positions': int((w_target > 0).sum()),
        }

    def _fallback_topk_portfolio(self, signals: pd.Series, topk: int = 30) -> pd.Series:
        """降级方案：TopK等权组合

        Args:
            signals: 信号
            topk: 持仓数量

        Returns:
            pd.Series: 等权组合权重
        """
        top_stocks = signals.nlargest(topk)
        weights = pd.Series(1/topk, index=top_stocks.index)
        return weights


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='组合优化工具')
    parser.add_argument('--market', default='cn', choices=['cn', 'hk'], help='市场代码')
    parser.add_argument('--signals', required=True, help='信号文件 (CSV)')
    parser.add_argument('--positions', default=None, help='当前持仓文件 (CSV)')
    parser.add_argument('--benchmark', default=None, help='基准权重文件 (CSV)')
    parser.add_argument('--output', default='orders.csv', help='订单输出文件')
    parser.add_argument('--total_value', type=float, default=100000000, help='总资产价值')

    args = parser.parse_args()

    # 加载信号
    signals = pd.read_csv(args.signals, index_col=0, squeeze=True)

    # 加载持仓
    if args.positions:
        positions_df = pd.read_csv(args.positions)
        current_positions = dict(zip(positions_df['stock'], positions_df['weight']))
    else:
        current_positions = {}

    # 加载基准
    if args.benchmark:
        benchmark_df = pd.read_csv(args.benchmark)
        benchmark_weights = dict(zip(benchmark_df['stock'], benchmark_df['weight']))
    else:
        benchmark_weights = None

    # 创建优化器
    optimizer = PortfolioOptimizer(market=args.market)

    # 生成优化方案
    plan = optimizer.generate_rebalance_plan(
        signals=signals,
        current_positions=current_positions,
        benchmark_weights=benchmark_weights,
        total_value=args.total_value
    )

    # 保存订单
    plan['orders'].to_csv(args.output, index=False)
    logger.info(f"订单已保存到: {args.output}")

    # 打印风险分析
    print("\n" + "=" * 60)
    print("风险分析报告")
    print("=" * 60)
    for key, value in plan['risk_analysis'].items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        elif isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == '__main__':
    main()

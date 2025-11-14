# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
回测评估系统

模拟真实交易环境，评估策略表现，包括：
- 事件驱动回测
- 交易成本模拟
- 绩效指标计算
- 风险指标分析
"""

from typing import Dict, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import qlib
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from loguru import logger


class BacktestEvaluator:
    """回测评估器

    使用 Qlib 回测框架验证策略表现

    Attributes:
        market (str): 市场代码
        benchmark (str): 基准指数

    Examples:
        >>> evaluator = BacktestEvaluator(market='cn')
        >>> results = evaluator.run_backtest(
        ...     signals=signals,
        ...     start_date='2024-01-01',
        ...     end_date='2024-11-14'
        ... )
        >>> print(f"夏普比率: {results['sharpe_ratio']:.2f}")
    """

    def __init__(
        self,
        market: str = 'cn',
        data_dir: str = '~/.qlib/qlib_data/cn_data',
        benchmark: str = 'SH000300'
    ):
        """初始化回测评估器

        Args:
            market: 市场代码
            data_dir: 数据目录
            benchmark: 基准指数
        """
        self.market = market
        self.benchmark = benchmark
        self.data_dir = data_dir

        # 初始化 qlib
        qlib.init(provider_uri=data_dir, region=market)

        # 交易成本配置
        self.exchange_config = {
            'freq': 'day',
            'limit_threshold': 0.095,  # 涨跌停限制 (A股10%)
            'deal_price': '$close',     # 成交价格
            'open_cost': 0.0005,        # 开仓成本 0.05% (佣金)
            'close_cost': 0.0015,       # 平仓成本 0.15% (佣金 + 印花税)
            'min_cost': 5,              # 最小手续费 5元
            'trade_unit': 100,          # 最小交易单位 (1手=100股)
        }

        logger.info(f"回测评估器初始化 [市场: {market}, 基准: {benchmark}]")

    def run_backtest(
        self,
        signals: pd.Series,
        start_date: str,
        end_date: str,
        topk: int = 30,
        init_cash: float = 100000000,  # 1亿
        verbose: bool = False
    ) -> Dict:
        """运行回测

        Args:
            signals: 交易信号
            start_date: 回测开始日期
            end_date: 回测结束日期
            topk: 持仓股票数量
            init_cash: 初始资金
            verbose: 是否打印详细信息

        Returns:
            Dict: 回测结果和绩效指标

        Examples:
            >>> results = evaluator.run_backtest(
            ...     signals=signals,
            ...     start_date='2024-01-01',
            ...     end_date='2024-11-14',
            ...     topk=30
            ... )
        """
        logger.info(f"开始回测 [{start_date} ~ {end_date}]")

        try:
            # 配置策略
            strategy = TopkDropoutStrategy(
                signal=signals,
                topk=topk,
                n_drop=5,  # 每次最多调整5只股票
                method_sell='bottom',
                method_buy='top',
            )

            # 配置执行器
            executor = SimulatorExecutor(
                time_per_step='day',
                verbose=verbose,
                track_data=True,
            )

            # 运行回测
            portfolio_metrics, indicator = backtest(
                strategy=strategy,
                executor=executor,
                start_time=start_date,
                end_time=end_date,
                account=init_cash,
                benchmark=self.benchmark,
                exchange_kwargs=self.exchange_config,
            )

            # 计算绩效指标
            performance = self._calculate_performance(
                portfolio_metrics,
                indicator,
                start_date,
                end_date
            )

            logger.success(
                f"✓ 回测完成 [收益: {performance['total_return']:.2%}, "
                f"夏普: {performance['sharpe_ratio']:.2f}, "
                f"最大回撤: {performance['max_drawdown']:.2%}]"
            )

            return performance

        except Exception as e:
            logger.error(f"回测失败: {str(e)}")
            raise

    def _calculate_performance(
        self,
        portfolio_metrics: pd.DataFrame,
        indicator: Dict,
        start_date: str,
        end_date: str
    ) -> Dict:
        """计算绩效指标

        Args:
            portfolio_metrics: 组合指标
            indicator: 回测指标
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            Dict: 绩效指标
        """
        # 提取关键指标
        returns = portfolio_metrics['return']
        bench_returns = portfolio_metrics.get('bench', pd.Series(0, index=returns.index))

        # 累计收益
        total_return = (1 + returns).prod() - 1
        bench_total_return = (1 + bench_returns).prod() - 1

        # 年化收益
        n_days = len(returns)
        n_years = n_days / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        bench_annual_return = (1 + bench_total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # 波动率 (年化)
        volatility = returns.std() * np.sqrt(252)

        # 夏普比率 (假设无风险利率 3%)
        risk_free_rate = 0.03
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0

        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # 信息比率 (相对基准)
        excess_returns = returns - bench_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = (annual_return - bench_annual_return) / tracking_error if tracking_error > 0 else 0

        # 胜率
        win_rate = (excess_returns > 0).sum() / len(excess_returns) if len(excess_returns) > 0 else 0

        # 换手率
        turnover = portfolio_metrics.get('turnover', pd.Series(0, index=returns.index)).mean()

        # 交易成本
        cost = portfolio_metrics.get('cost', pd.Series(0, index=returns.index)).sum()

        return {
            # 收益指标
            'total_return': total_return,
            'annual_return': annual_return,
            'bench_total_return': bench_total_return,
            'bench_annual_return': bench_annual_return,
            'excess_return': total_return - bench_total_return,
            'annual_excess_return': annual_return - bench_annual_return,

            # 风险指标
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,

            # 相对指标
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'win_rate': win_rate,

            # 交易指标
            'turnover': turnover,
            'total_cost': cost,

            # 时间信息
            'start_date': start_date,
            'end_date': end_date,
            'n_trading_days': n_days,

            # 原始数据
            'portfolio_metrics': portfolio_metrics,
        }


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='回测评估工具')
    parser.add_argument('--market', default='cn', choices=['cn', 'hk'], help='市场代码')
    parser.add_argument('--data_dir', default='~/.qlib/qlib_data/cn_data', help='数据目录')
    parser.add_argument('--signals', required=True, help='信号文件 (CSV)')
    parser.add_argument('--start_date', required=True, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end_date', required=True, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--topk', type=int, default=30, help='持仓股票数')
    parser.add_argument('--init_cash', type=float, default=100000000, help='初始资金')
    parser.add_argument('--output', default='backtest_results.csv', help='结果输出文件')

    args = parser.parse_args()

    # 加载信号
    signals = pd.read_csv(args.signals, index_col=0, squeeze=True)

    # 创建回测器
    evaluator = BacktestEvaluator(
        market=args.market,
        data_dir=args.data_dir
    )

    # 运行回测
    results = evaluator.run_backtest(
        signals=signals,
        start_date=args.start_date,
        end_date=args.end_date,
        topk=args.topk,
        init_cash=args.init_cash,
        verbose=True
    )

    # 保存结果
    results['portfolio_metrics'].to_csv(args.output)
    logger.info(f"回测结果已保存到: {args.output}")

    # 打印绩效报告
    print("\n" + "=" * 60)
    print("回测绩效报告")
    print("=" * 60)
    print(f"回测区间: {results['start_date']} ~ {results['end_date']}")
    print(f"交易天数: {results['n_trading_days']}")
    print("\n收益指标:")
    print(f"  总收益率: {results['total_return']:.2%}")
    print(f"  年化收益: {results['annual_return']:.2%}")
    print(f"  基准收益: {results['bench_total_return']:.2%}")
    print(f"  超额收益: {results['excess_return']:.2%}")
    print("\n风险指标:")
    print(f"  年化波动: {results['volatility']:.2%}")
    print(f"  夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {results['max_drawdown']:.2%}")
    print("\n相对指标:")
    print(f"  信息比率: {results['information_ratio']:.2f}")
    print(f"  跟踪误差: {results['tracking_error']:.2%}")
    print(f"  胜率: {results['win_rate']:.2%}")
    print("\n交易指标:")
    print(f"  平均换手: {results['turnover']:.2%}")
    print(f"  总成本: ¥{results['total_cost']:,.2f}")


if __name__ == '__main__':
    main()

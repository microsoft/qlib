# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
自动化交易主控制器

整合所有模块，实现完整的自动化交易流程：
数据更新 → 信号生成 → 组合优化 → 回测评估 → 报告生成
"""

import sys
import yaml
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger

# 导入自定义模块
from data_updater import MultiMarketDataUpdater
from signal_generator import SignalGenerator
from portfolio_optimizer import PortfolioOptimizer
from backtester import BacktestEvaluator
from report_generator import ReportGenerator


class AutoTradingController:
    """自动化交易主控制器

    协调所有模块，执行完整的每日自动化流程

    Attributes:
        config (Dict): 系统配置
        modules (Dict): 各功能模块实例

    Examples:
        >>> controller = AutoTradingController(config_path='config.yaml')
        >>> result = controller.run_daily_pipeline()
        >>> print(result['success'])
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """初始化主控制器

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)

        logger.info("=" * 70)
        logger.info("自动化交易系统启动")
        logger.info("=" * 70)

        # 初始化所有模块
        self.modules = {}

        try:
            # 数据更新模块
            self.modules['data_updater'] = MultiMarketDataUpdater(
                base_dir=self.config['data']['base_dir']
            )

            # 信号生成模块
            self.modules['signal_generator'] = SignalGenerator(
                market=self.config['market'],
                data_dir=self._get_data_dir(),
                models_dir=self.config['models']['models_dir'],
                benchmark=self.config['benchmark']
            )

            # 组合优化模块
            self.modules['portfolio_optimizer'] = PortfolioOptimizer(
                market=self.config['market'],
                lamb=self.config['risk_control']['lambda'],
                max_turnover=self.config['risk_control']['max_turnover'],
                max_position_deviation=self.config['risk_control']['max_position_deviation'],
            )

            # 回测评估模块
            self.modules['backtester'] = BacktestEvaluator(
                market=self.config['market'],
                data_dir=self._get_data_dir(),
                benchmark=self.config['benchmark']
            )

            # 报告生成模块
            self.modules['reporter'] = ReportGenerator(
                output_dir=self.config['report']['output_dir']
            )

            logger.success("✓ 所有模块初始化完成")

        except Exception as e:
            logger.error(f"✗ 模块初始化失败: {str(e)}")
            raise

    def run_daily_pipeline(
        self,
        date: Optional[str] = None,
        skip_data_update: bool = False,
        skip_backtest: bool = False,
        force_retrain: bool = False
    ) -> Dict:
        """运行每日自动化流程

        Args:
            date: 运行日期 (YYYY-MM-DD)，默认为今天
            skip_data_update: 是否跳过数据更新
            skip_backtest: 是否跳过回测
            force_retrain: 是否强制重新训练模型

        Returns:
            Dict: 运行结果

        Examples:
            >>> result = controller.run_daily_pipeline(date='2024-11-14')
            >>> if result['success']:
            ...     print(f"报告: {result['report_files']['html_report']}")
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"开始执行 {date} 的自动化交易流程")
        logger.info("=" * 70)
        logger.info("")

        start_time = datetime.now()
        results = {}

        try:
            # ========== 步骤 1: 数据更新 ==========
            if not skip_data_update:
                logger.info("【步骤 1/5】 更新市场数据")
                logger.info("-" * 70)

                data_update_result = self.modules['data_updater'].update_market_data(
                    market=self.config['market'],
                    end_date=date
                )

                results['data_update'] = data_update_result

                if not data_update_result['success']:
                    raise RuntimeError(f"数据更新失败: {data_update_result['error']}")

                # 数据质量检查
                quality_report = self.modules['data_updater'].verify_data_quality(
                    market=self.config['market']
                )
                results['data_quality'] = quality_report

                logger.info("")
            else:
                logger.warning("跳过数据更新步骤")

            # ========== 步骤 2: 信号生成 ==========
            logger.info("【步骤 2/5】 生成交易信号")
            logger.info("-" * 70)

            signals, signal_quality = self.modules['signal_generator'].generate_signals(
                date=date,
                force_retrain=force_retrain
            )

            results['signals'] = signals
            results['signal_quality'] = signal_quality

            logger.info(f"✓ 生成 {len(signals)} 个交易信号")
            logger.info("")

            # ========== 步骤 3: 组合优化 ==========
            logger.info("【步骤 3/5】 优化投资组合")
            logger.info("-" * 70)

            # 加载当前持仓
            current_positions = self._load_current_positions()

            # 加载基准权重
            benchmark_weights = self._load_benchmark_weights()

            # 生成再平衡计划
            rebalance_plan = self.modules['portfolio_optimizer'].generate_rebalance_plan(
                signals=signals,
                current_positions=current_positions,
                benchmark_weights=benchmark_weights,
                constraints=self.config.get('constraints', None),
                total_value=self.config['portfolio']['total_value']
            )

            results['rebalance_plan'] = rebalance_plan

            logger.info(f"✓ 生成 {len(rebalance_plan['orders'])} 个交易订单")
            logger.info(f"  换手率: {rebalance_plan['turnover']:.2%}")
            logger.info("")

            # ========== 步骤 4: 回测评估 ==========
            if not skip_backtest:
                logger.info("【步骤 4/5】 回测验证策略")
                logger.info("-" * 70)

                # 回测最近60天
                backtest_start = (pd.Timestamp(date) - timedelta(days=90)).strftime('%Y-%m-%d')

                backtest_results = self.modules['backtester'].run_backtest(
                    signals=signals,
                    start_date=backtest_start,
                    end_date=date,
                    topk=self.config['strategy']['topk'],
                    init_cash=self.config['portfolio']['total_value']
                )

                results['backtest'] = backtest_results

                logger.info(f"✓ 回测完成")
                logger.info(f"  年化收益: {backtest_results['annual_return']:.2%}")
                logger.info(f"  夏普比率: {backtest_results['sharpe_ratio']:.2f}")
                logger.info("")
            else:
                logger.warning("跳过回测步骤")
                # 使用默认值
                results['backtest'] = self._get_default_backtest_results()

            # ========== 步骤 5: 生成报告 ==========
            logger.info("【步骤 5/5】 生成交易报告")
            logger.info("-" * 70)

            report_files = self.modules['reporter'].generate_daily_report(
                date=date,
                signals=signals,
                rebalance_plan=rebalance_plan,
                backtest_results=results['backtest'],
                risk_analysis=rebalance_plan['risk_analysis'],
                signal_quality=signal_quality
            )

            results['report_files'] = report_files

            logger.info(f"✓ 报告已生成")
            logger.info(f"  HTML: {report_files['html_report']}")
            logger.info(f"  Excel: {report_files['excel_report']}")
            logger.info("")

            # ========== 完成 ==========
            elapsed_time = (datetime.now() - start_time).total_seconds()

            logger.info("=" * 70)
            logger.success(f"✅ 自动化流程执行成功！ [耗时: {elapsed_time:.1f}s]")
            logger.info("=" * 70)

            return {
                'success': True,
                'date': date,
                'elapsed_time': elapsed_time,
                'results': results,
                'summary': {
                    'signals_count': len(signals),
                    'orders_count': len(rebalance_plan['orders']),
                    'turnover': rebalance_plan['turnover'],
                    'sharpe': results['backtest']['sharpe_ratio'],
                    'annual_return': results['backtest']['annual_return'],
                }
            }

        except Exception as e:
            elapsed_time = (datetime.now() - start_time).total_seconds()

            logger.error("")
            logger.error("=" * 70)
            logger.error(f"❌ 自动化流程执行失败！ [耗时: {elapsed_time:.1f}s]")
            logger.error(f"错误: {str(e)}")
            logger.error("=" * 70)

            import traceback
            logger.error(traceback.format_exc())

            return {
                'success': False,
                'date': date,
                'elapsed_time': elapsed_time,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def _load_config(self, path: str) -> Dict:
        """加载配置文件

        Args:
            path: 配置文件路径

        Returns:
            Dict: 配置字典
        """
        config_file = Path(path)

        if not config_file.exists():
            # 使用默认配置
            logger.warning(f"配置文件不存在: {path}，使用默认配置")
            return self._get_default_config()

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            logger.info(f"配置文件加载成功: {path}")
            return config

        except Exception as e:
            logger.error(f"配置文件加载失败: {str(e)}")
            raise

    def _get_default_config(self) -> Dict:
        """获取默认配置

        Returns:
            Dict: 默认配置
        """
        return {
            'market': 'cn',
            'benchmark': 'SH000300',
            'data': {
                'base_dir': '~/.qlib/qlib_data',
            },
            'models': {
                'models_dir': './models',
            },
            'strategy': {
                'topk': 30,
            },
            'risk_control': {
                'lambda': 1.0,
                'max_turnover': 0.3,
                'max_position_deviation': 0.05,
            },
            'portfolio': {
                'total_value': 100000000,
            },
            'report': {
                'output_dir': './reports',
            },
        }

    def _get_data_dir(self) -> str:
        """获取数据目录

        Returns:
            str: 数据目录路径
        """
        base_dir = Path(self.config['data']['base_dir']).expanduser()
        market = self.config['market']
        return str(base_dir / f'{market}_data')

    def _load_current_positions(self) -> Dict[str, float]:
        """加载当前持仓

        Returns:
            Dict[str, float]: 持仓字典 {股票代码: 权重}
        """
        # 从配置或文件加载
        positions_file = self.config.get('portfolio', {}).get('positions_file', None)

        if positions_file and Path(positions_file).exists():
            df = pd.read_csv(positions_file)
            return dict(zip(df['stock'], df['weight']))
        else:
            # 返回空持仓（从零开始）
            logger.warning("未找到持仓文件，从零持仓开始")
            return {}

    def _load_benchmark_weights(self) -> Optional[Dict[str, float]]:
        """加载基准权重

        Returns:
            Optional[Dict[str, float]]: 基准权重字典，None表示使用等权
        """
        # 从配置或文件加载
        benchmark_file = self.config.get('benchmark_weights_file', None)

        if benchmark_file and Path(benchmark_file).exists():
            df = pd.read_csv(benchmark_file)
            return dict(zip(df['stock'], df['weight']))
        else:
            # 返回None，使用等权基准
            return None

    def _get_default_backtest_results(self) -> Dict:
        """获取默认的回测结果（当跳过回测时使用）

        Returns:
            Dict: 默认回测结果
        """
        return {
            'total_return': 0,
            'annual_return': 0,
            'bench_total_return': 0,
            'bench_annual_return': 0,
            'excess_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'information_ratio': 0,
            'tracking_error': 0,
            'win_rate': 0,
            'turnover': 0,
            'total_cost': 0,
        }


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description='A股/港股/ETF 自动化交易系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行今天的流程
  python main_controller.py --config config.yaml

  # 指定日期运行
  python main_controller.py --config config.yaml --date 2024-11-14

  # 跳过数据更新（使用现有数据）
  python main_controller.py --skip-data-update

  # 强制重新训练模型
  python main_controller.py --force-retrain
        """
    )

    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--date', default=None, help='运行日期 (YYYY-MM-DD)')
    parser.add_argument('--skip-data-update', action='store_true', help='跳过数据更新')
    parser.add_argument('--skip-backtest', action='store_true', help='跳过回测')
    parser.add_argument('--force-retrain', action='store_true', help='强制重新训练模型')
    parser.add_argument('--verbose', action='store_true', help='详细输出')

    args = parser.parse_args()

    # 配置日志级别
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    # 初始化控制器
    try:
        controller = AutoTradingController(config_path=args.config)
    except Exception as e:
        logger.error(f"控制器初始化失败: {str(e)}")
        sys.exit(1)

    # 运行流程
    result = controller.run_daily_pipeline(
        date=args.date,
        skip_data_update=args.skip_data_update,
        skip_backtest=args.skip_backtest,
        force_retrain=args.force_retrain
    )

    # 输出结果
    if result['success']:
        print("\n" + "=" * 70)
        print("📊 执行摘要")
        print("=" * 70)
        print(f"日期: {result['date']}")
        print(f"信号数: {result['summary']['signals_count']}")
        print(f"订单数: {result['summary']['orders_count']}")
        print(f"换手率: {result['summary']['turnover']:.2%}")
        print(f"年化收益: {result['summary']['annual_return']:.2%}")
        print(f"夏普比率: {result['summary']['sharpe']:.2f}")
        print(f"耗时: {result['elapsed_time']:.1f}秒")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("❌ 执行失败")
        print("=" * 70)
        print(f"错误: {result['error']}")
        print("=" * 70)
        sys.exit(1)


if __name__ == '__main__':
    main()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
A股/港股/ETF 自动化交易系统

此模块提供完整的量化交易自动化解决方案，包括：
- 多市场数据自动更新
- 智能信号生成（滚动训练+多模型集成）
- 投资组合优化（目标权重+订单生成）
- 回测评估与风险分析
- 自动化报告生成

使用示例：
    from scripts.auto_trading.main_controller import AutoTradingController

    controller = AutoTradingController(config_path='config.yaml')
    result = controller.run_daily_pipeline()
"""

__version__ = "1.0.0"
__author__ = "Auto Trading Team"

from .data_updater import MultiMarketDataUpdater
from .signal_generator import SignalGenerator
from .portfolio_optimizer import PortfolioOptimizer
from .backtester import BacktestEvaluator
from .report_generator import ReportGenerator
from .main_controller import AutoTradingController

__all__ = [
    'MultiMarketDataUpdater',
    'SignalGenerator',
    'PortfolioOptimizer',
    'BacktestEvaluator',
    'ReportGenerator',
    'AutoTradingController',
]

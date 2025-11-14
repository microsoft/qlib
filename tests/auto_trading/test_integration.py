# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
自动化交易系统集成测试
"""

import sys
import unittest
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts' / 'auto_trading'))


class TestAutoTradingIntegration(unittest.TestCase):
    """集成测试（需要数据）"""

    def test_imports(self):
        """测试模块导入"""
        try:
            from data_updater import MultiMarketDataUpdater
            from signal_generator import SignalGenerator
            from portfolio_optimizer import PortfolioOptimizer
            from backtester import BacktestEvaluator
            from report_generator import ReportGenerator
            from main_controller import AutoTradingController
        except ImportError as e:
            self.fail(f"模块导入失败: {str(e)}")

    def test_config_loading(self):
        """测试配置文件加载"""
        from main_controller import AutoTradingController

        controller = AutoTradingController(config_path='non_existent.yaml')
        self.assertIsNotNone(controller.config)
        self.assertIn('market', controller.config)


if __name__ == '__main__':
    unittest.main()

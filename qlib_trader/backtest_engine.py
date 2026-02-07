"""
回测引擎模块
============
负责策略回测和绩效分析。
"""

import time
from typing import Any, Dict, Optional, Tuple

from qlib_trader.config import (
    STRATEGY_PRESETS,
    TIME_PRESETS,
    REGIONS,
    ConfigBuilder,
)
from qlib_trader.utils import (
    Style,
    confirm,
    format_number,
    get_choice,
    print_menu,
    print_section,
    print_table,
)


class BacktestEngine:
    """回测引擎"""

    def run_menu(self, builder: ConfigBuilder):
        """回测管理菜单"""
        while True:
            print_menu("回测管理", [
                "查看可用策略",
                "选择策略",
                "运行回测",
                "修改回测参数",
            ])
            choice = get_choice("请选择", 4)
            if choice == 0:
                return
            elif choice == 1:
                self.show_strategies()
            elif choice == 2:
                self.select_strategy(builder)
            elif choice == 3:
                self.run_backtest(builder)
            elif choice == 4:
                self.configure_backtest(builder)

    def show_strategies(self):
        """显示可用策略"""
        print_section("可用交易策略")
        rows = []
        for key, conf in STRATEGY_PRESETS.items():
            rows.append([
                key,
                conf["name"],
                f"TopK={conf['kwargs'].get('topk', '-')}, "
                f"Drop={conf['kwargs'].get('n_drop', '-')}",
                conf["description"],
            ])
        print_table(
            ["标识", "名称", "参数", "说明"],
            rows,
            [15, 20, 18, 35],
        )

    def select_strategy(self, builder: ConfigBuilder):
        """选择策略"""
        print_section("选择策略")
        keys = list(STRATEGY_PRESETS.keys())
        options = [f"{v['name']} - {v['description']}" for v in STRATEGY_PRESETS.values()]
        print_menu("可选策略", options, show_back=True)
        choice = get_choice("请选择策略", len(options))
        if choice == 0:
            return
        strategy_key = keys[choice - 1]
        builder.set_strategy(strategy_key)
        print(Style.success(f"\n  已选择策略: {STRATEGY_PRESETS[strategy_key]['name']}"))

    def configure_backtest(self, builder: ConfigBuilder):
        """配置回测参数"""
        print_section("回测参数配置")
        from qlib_trader.utils import get_input

        # Account
        current_account = format_number(builder.account)
        new_account = get_input(f"初始资金 (当前: {current_account})", str(builder.account))
        try:
            builder.set_account(int(float(new_account)))
        except ValueError:
            print(Style.warning("  无效金额，保持原值"))

        # Strategy TopK
        strategy_conf = STRATEGY_PRESETS[builder.strategy_key]
        current_topk = strategy_conf["kwargs"].get("topk", 50)
        new_topk = get_input(f"持仓股票数 TopK (当前: {current_topk})", str(current_topk))
        try:
            strategy_conf["kwargs"]["topk"] = int(new_topk)
        except ValueError:
            pass

        # Strategy n_drop
        current_drop = strategy_conf["kwargs"].get("n_drop", 5)
        new_drop = get_input(f"每期淘汰数 n_drop (当前: {current_drop})", str(current_drop))
        try:
            strategy_conf["kwargs"]["n_drop"] = int(new_drop)
        except ValueError:
            pass

        print(Style.success("\n  回测参数已更新"))

    def run_backtest(
        self,
        builder: ConfigBuilder,
        model=None,
        dataset=None,
        recorder=None,
    ) -> Optional[Dict]:
        """运行回测

        Args:
            builder: 配置构建器
            model: 已训练模型（可选，如未提供则从recorder加载）
            dataset: 数据集（可选）
            recorder: 记录器（可选）

        Returns:
            回测结果字典或None
        """
        print_section("运行回测")
        strategy_conf = STRATEGY_PRESETS[builder.strategy_key]
        region_conf = REGIONS[builder.region]
        time_conf = TIME_PRESETS[builder.time_key]

        print(f"  策略      : {strategy_conf['name']}")
        print(f"  回测区间  : {time_conf['test'][0]} ~ {time_conf['test'][1]}")
        print(f"  基准指数  : {region_conf['benchmark']}")
        print(f"  初始资金  : {format_number(builder.account)}")
        print(f"  交易费用  : 买入 {region_conf['open_cost']*100:.2f}% / "
              f"卖出 {region_conf['close_cost']*100:.2f}%")
        print()

        if model is None or dataset is None:
            print(Style.warning("  需要先训练模型才能运行回测"))
            if not confirm("是否先训练模型？"):
                return None
            from qlib_trader.model_manager import ModelManager
            result = ModelManager().train_model(builder)
            if result is None:
                return None
            model, dataset, recorder = result

        if not confirm("确认运行回测？"):
            return None

        try:
            from qlib.workflow import R
            from qlib.workflow.record_temp import PortAnaRecord

            print(Style.info("\n  正在运行回测..."))
            start_time = time.time()

            backtest_config = builder.build_backtest_config()

            # Inject signal into strategy
            backtest_config["strategy"]["kwargs"]["signal"] = (model, dataset)

            if recorder is None:
                recorder = R.get_recorder()

            par = PortAnaRecord(recorder, backtest_config, "day")
            par.generate()

            elapsed = time.time() - start_time
            print(Style.success(f"  回测完成 (耗时 {elapsed:.1f}s)"))

            # Display results
            results = self._display_results(recorder)
            return results

        except Exception as e:
            print(Style.error(f"\n  回测失败: {e}"))
            import traceback
            traceback.print_exc()
            return None

    def _display_results(self, recorder) -> Dict:
        """显示回测结果"""
        results = {}
        try:
            portfolio_analysis = recorder.load_object("portfolio_analysis")
        except Exception:
            print(Style.info("  (请查看 MLflow 获取详细回测结果)"))
            return results

        print_section("回测绩效报告")

        try:
            # Extract key metrics
            if hasattr(portfolio_analysis, "items"):
                report_normal = portfolio_analysis.get("report_normal", None)
                indicator_normal = portfolio_analysis.get("indicator_normal", None)

                if report_normal is not None and hasattr(report_normal, "iloc"):
                    excess = report_normal.get("excess_return_without_cost", None)
                    if excess is not None:
                        import pandas as pd
                        metrics = self._compute_metrics(report_normal)
                        results = metrics
                        self._print_metrics(metrics)
                    else:
                        print(f"  报告内容: {list(report_normal.columns) if hasattr(report_normal, 'columns') else report_normal}")

                if indicator_normal is not None:
                    self._print_indicators(indicator_normal)
            else:
                print(f"  {portfolio_analysis}")

        except Exception as e:
            print(Style.warning(f"  结果解析异常: {e}"))
            print(Style.info("  原始结果已保存到 MLflow 实验记录中"))

        return results

    def _compute_metrics(self, report_df) -> Dict[str, float]:
        """从回测报告计算关键指标"""
        import numpy as np

        metrics = {}
        try:
            # Cumulative returns
            if "return" in report_df.columns:
                returns = report_df["return"]
                cum_return = (1 + returns).prod() - 1
                metrics["累计收益率"] = cum_return

                # Annualized return
                n_days = len(returns)
                ann_return = (1 + cum_return) ** (252 / max(n_days, 1)) - 1
                metrics["年化收益率"] = ann_return

                # Sharpe ratio
                if returns.std() > 0:
                    sharpe = returns.mean() / returns.std() * np.sqrt(252)
                    metrics["夏普比率"] = sharpe

                # Max drawdown
                cum_returns = (1 + returns).cumprod()
                peak = cum_returns.expanding(min_periods=1).max()
                drawdown = (cum_returns / peak) - 1
                max_dd = drawdown.min()
                metrics["最大回撤"] = max_dd

                # Calmar ratio
                if max_dd != 0:
                    metrics["卡尔马比率"] = ann_return / abs(max_dd)

                # Sortino ratio
                downside = returns[returns < 0]
                if len(downside) > 0 and downside.std() > 0:
                    metrics["索提诺比率"] = returns.mean() / downside.std() * np.sqrt(252)

            # Excess returns
            if "excess_return_without_cost" in report_df.columns:
                excess = report_df["excess_return_without_cost"]
                metrics["超额累计收益(无费)"] = (1 + excess).prod() - 1

            if "excess_return_with_cost" in report_df.columns:
                excess = report_df["excess_return_with_cost"]
                metrics["超额累计收益(含费)"] = (1 + excess).prod() - 1

        except Exception:
            pass

        return metrics

    def _print_metrics(self, metrics: Dict[str, float]):
        """打印关键指标"""
        print(Style.bold("\n  关键绩效指标:"))
        print("  " + "-" * 45)
        for name, value in metrics.items():
            if "收益" in name or "回撤" in name:
                formatted = f"{value:.2%}"
                if value > 0:
                    formatted = Style.success(formatted)
                elif value < 0:
                    formatted = Style.error(formatted)
            else:
                formatted = f"{value:.4f}"
                if "夏普" in name or "卡尔马" in name or "索提诺" in name:
                    if value > 1:
                        formatted = Style.success(formatted)
                    elif value < 0:
                        formatted = Style.error(formatted)
            print(f"  {name:<20s} : {formatted}")
        print("  " + "-" * 45)

    def _print_indicators(self, indicator):
        """打印交易指标"""
        try:
            if hasattr(indicator, "items"):
                print(Style.bold("\n  交易指标:"))
                print("  " + "-" * 45)
                for name, value in indicator.items():
                    if isinstance(value, (int, float)):
                        print(f"  {name:<20s} : {value:.4f}")
                    elif hasattr(value, "mean"):
                        print(f"  {name:<20s} : {value.mean():.4f} (mean)")
        except Exception:
            pass

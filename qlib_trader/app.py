"""
QLib Trader 主应用
==================
交互式量化交易集成平台入口。
"""

import argparse
import sys

from qlib_trader.config import (
    ConfigBuilder,
    DATASET_HANDLERS,
    MODEL_PRESETS,
    REGIONS,
    STRATEGY_PRESETS,
    TIME_PRESETS,
)
from qlib_trader.data_manager import DataManager
from qlib_trader.model_manager import ModelManager
from qlib_trader.backtest_engine import BacktestEngine
from qlib_trader.pipeline import Pipeline
from qlib_trader.utils import (
    Style,
    confirm,
    get_choice,
    print_banner,
    print_menu,
    print_section,
)


class QLibTraderApp:
    """QLib Trader 主应用"""

    def __init__(self):
        self.builder = ConfigBuilder()
        self.data_mgr = DataManager()
        self.model_mgr = ModelManager()
        self.backtest_eng = BacktestEngine()
        self.pipeline = Pipeline()

    def run_interactive(self):
        """运行交互式菜单"""
        print_banner()
        print(Style.info("  欢迎使用 QLib Trader 量化交易集成平台！"))
        print(Style.info("  输入数字选择功能，输入 0 返回上一级\n"))

        while True:
            print_menu("主菜单", [
                "一键流水线 (推荐)",
                "数据管理",
                "模型管理",
                "回测管理",
                "查看/修改配置",
                "退出",
            ], show_back=False)

            choice = get_choice("请选择功能", 6, allow_zero=False)

            if choice == 1:
                self.pipeline.run_menu(self.builder)
            elif choice == 2:
                self.data_mgr.run_menu()
            elif choice == 3:
                self.model_mgr.run_menu(self.builder)
            elif choice == 4:
                self.backtest_eng.run_menu(self.builder)
            elif choice == 5:
                self._config_menu()
            elif choice == 6:
                if confirm("确认退出？"):
                    print(Style.info("\n  感谢使用 QLib Trader，再见！\n"))
                    break

    def _config_menu(self):
        """配置查看/修改菜单"""
        while True:
            print_section("配置管理")
            print(self.builder.summary())
            print()
            print_menu("修改配置", [
                "修改市场区域",
                "修改AI模型",
                "修改数据集/因子",
                "修改交易策略",
                "修改时间段",
                "修改初始资金",
                "重置为默认",
            ])
            choice = get_choice("请选择", 7)
            if choice == 0:
                return
            elif choice == 1:
                self._select_from_dict("市场区域", REGIONS, self.builder.set_region, "name")
            elif choice == 2:
                self._select_from_dict("AI模型", MODEL_PRESETS, self.builder.set_model, "name")
            elif choice == 3:
                self._select_from_dict("数据集", DATASET_HANDLERS, self.builder.set_handler, "name")
            elif choice == 4:
                self._select_from_dict("交易策略", STRATEGY_PRESETS, self.builder.set_strategy, "name")
            elif choice == 5:
                self._select_from_dict("时间段", TIME_PRESETS, self.builder.set_time, "name")
            elif choice == 6:
                from qlib_trader.utils import get_input
                new_val = get_input("初始资金", str(self.builder.account))
                try:
                    self.builder.set_account(int(float(new_val)))
                    print(Style.success("  已更新"))
                except ValueError:
                    print(Style.warning("  无效金额"))
            elif choice == 7:
                self.builder = ConfigBuilder()
                print(Style.success("  已重置为默认配置"))

    def _select_from_dict(self, title, options_dict, setter_fn, display_key):
        """通用选择器"""
        keys = list(options_dict.keys())
        options = [f"{v[display_key]}" for v in options_dict.values()]
        print_menu(f"选择{title}", options, show_back=True)
        choice = get_choice("请选择", len(options))
        if choice == 0:
            return
        try:
            setter_fn(keys[choice - 1])
            print(Style.success(f"  已设置{title}: {options[choice - 1]}"))
        except ValueError as e:
            print(Style.error(f"  设置失败: {e}"))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="QLib Trader - 量化交易集成平台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python -m qlib_trader                             # 启动交互式菜单
  python -m qlib_trader pipeline                    # 使用默认配置运行流水线
  python -m qlib_trader pipeline --model xgboost    # 指定模型
  python -m qlib_trader data --status               # 查看数据状态
  python -m qlib_trader data --download cn          # 下载中国市场数据
  python -m qlib_trader train --model lightgbm      # 训练模型
  python -m qlib_trader backtest                    # 运行回测
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # Pipeline command
    pipe_parser = subparsers.add_parser("pipeline", help="运行一键流水线")
    pipe_parser.add_argument("--model", choices=list(MODEL_PRESETS.keys()),
                             default="lightgbm", help="AI模型 (默认: lightgbm)")
    pipe_parser.add_argument("--handler", choices=list(DATASET_HANDLERS.keys()),
                             default="alpha158", help="数据集 (默认: alpha158)")
    pipe_parser.add_argument("--strategy", choices=list(STRATEGY_PRESETS.keys()),
                             default="topk_dropout", help="策略 (默认: topk_dropout)")
    pipe_parser.add_argument("--region", choices=list(REGIONS.keys()),
                             default="cn", help="市场 (默认: cn)")
    pipe_parser.add_argument("--time", choices=list(TIME_PRESETS.keys()),
                             default="default", help="时间段 (默认: default)")
    pipe_parser.add_argument("--account", type=int, default=100_000_000,
                             help="初始资金 (默认: 100000000)")

    # Data command
    data_parser = subparsers.add_parser("data", help="数据管理")
    data_parser.add_argument("--status", action="store_true", help="查看数据状态")
    data_parser.add_argument("--download", choices=list(REGIONS.keys()), help="下载数据")
    data_parser.add_argument("--check", choices=list(REGIONS.keys()), help="检查数据完整性")

    # Train command
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--model", choices=list(MODEL_PRESETS.keys()),
                              default="lightgbm", help="模型")
    train_parser.add_argument("--handler", choices=list(DATASET_HANDLERS.keys()),
                              default="alpha158", help="数据集")
    train_parser.add_argument("--region", choices=list(REGIONS.keys()),
                              default="cn", help="市场")
    train_parser.add_argument("--time", choices=list(TIME_PRESETS.keys()),
                              default="default", help="时间段")

    # Backtest command
    bt_parser = subparsers.add_parser("backtest", help="运行回测")
    bt_parser.add_argument("--strategy", choices=list(STRATEGY_PRESETS.keys()),
                           default="topk_dropout", help="策略")
    bt_parser.add_argument("--model", choices=list(MODEL_PRESETS.keys()),
                           default="lightgbm", help="模型")
    bt_parser.add_argument("--region", choices=list(REGIONS.keys()),
                           default="cn", help="市场")

    # Models command
    subparsers.add_parser("models", help="查看可用模型列表")

    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()

    if args.command is None:
        # Interactive mode
        app = QLibTraderApp()
        try:
            app.run_interactive()
        except KeyboardInterrupt:
            print(Style.info("\n\n  已退出 QLib Trader\n"))
            sys.exit(0)

    elif args.command == "pipeline":
        builder = ConfigBuilder()
        builder.set_region(args.region)
        builder.set_model(args.model)
        builder.set_handler(args.handler)
        builder.set_strategy(args.strategy)
        builder.set_time(args.time)
        builder.set_account(args.account)

        print_banner()
        print(builder.summary())
        pipeline = Pipeline()
        pipeline.run_quick(builder)

    elif args.command == "data":
        mgr = DataManager()
        if args.status:
            mgr.show_data_status()
        elif args.download:
            mgr.download_data(args.download)
        elif args.check:
            mgr.check_data_health(args.check)
        else:
            mgr.run_menu()

    elif args.command == "train":
        builder = ConfigBuilder()
        builder.set_region(args.region)
        builder.set_model(args.model)
        builder.set_handler(args.handler)
        builder.set_time(args.time)

        print_banner()
        ModelManager().train_model(builder)

    elif args.command == "backtest":
        builder = ConfigBuilder()
        builder.set_region(args.region)
        builder.set_model(args.model)
        builder.set_strategy(args.strategy)

        print_banner()
        BacktestEngine().run_backtest(builder)

    elif args.command == "models":
        print_banner()
        ModelManager().show_available_models()


if __name__ == "__main__":
    main()

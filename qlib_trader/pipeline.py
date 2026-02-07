"""
一键流水线模块
==============
将数据 → 训练 → 回测 → 报告整合为一键流程。
"""

import time
from typing import Optional

from qlib_trader.config import (
    DATASET_HANDLERS,
    MODEL_PRESETS,
    REGIONS,
    STRATEGY_PRESETS,
    TIME_PRESETS,
    ConfigBuilder,
)
from qlib_trader.data_manager import DataManager
from qlib_trader.model_manager import ModelManager
from qlib_trader.backtest_engine import BacktestEngine
from qlib_trader.utils import (
    Style,
    check_data_exists,
    confirm,
    format_number,
    get_choice,
    print_menu,
    print_section,
)


class Pipeline:
    """一键流水线"""

    def __init__(self):
        self.data_mgr = DataManager()
        self.model_mgr = ModelManager()
        self.backtest_eng = BacktestEngine()

    def run_menu(self, builder: ConfigBuilder):
        """流水线菜单"""
        while True:
            print_menu("一键流水线", [
                "快速入门 (推荐新手)",
                "自定义流水线",
                "使用预设模板",
            ])
            choice = get_choice("请选择", 3)
            if choice == 0:
                return
            elif choice == 1:
                self.quick_start(builder)
            elif choice == 2:
                self.custom_pipeline(builder)
            elif choice == 3:
                self.template_pipeline(builder)

    def quick_start(self, builder: ConfigBuilder):
        """快速入门 - 使用默认配置一键运行"""
        print_section("快速入门流水线")
        print("""
  本流程将自动执行以下步骤：
    1. 检查/下载数据
    2. 使用 LightGBM + Alpha158 训练模型
    3. 使用 TopK-Dropout 策略回测
    4. 生成绩效报告

  默认配置：
    - 市场: 中国A股 (沪深300)
    - 模型: LightGBM (快速稳定)
    - 因子: Alpha158 (158个经典因子)
    - 策略: TopK50-Drop5 (持有50只，每期换5只)
    - 回测: 2017-01-01 至 2020-08-01
    - 资金: 1亿元
        """)

        if not confirm("确认使用默认配置运行？"):
            return

        # Reset to defaults
        builder.set_region("cn")
        builder.set_model("lightgbm")
        builder.set_handler("alpha158")
        builder.set_strategy("topk_dropout")
        builder.set_time("default")
        builder.set_account(100_000_000)

        self._run_pipeline(builder)

    def custom_pipeline(self, builder: ConfigBuilder):
        """自定义流水线 - 逐步选择配置"""
        print_section("自定义流水线配置")

        # Step 1: Region
        print(Style.bold("\n  [步骤 1/5] 选择市场"))
        keys = list(REGIONS.keys())
        options = [f"{v['name']}" for v in REGIONS.values()]
        print_menu("市场", options, show_back=True)
        choice = get_choice("请选择", len(options))
        if choice == 0:
            return
        builder.set_region(keys[choice - 1])

        # Step 2: Model
        print(Style.bold("\n  [步骤 2/5] 选择AI模型"))
        keys = list(MODEL_PRESETS.keys())
        options = [f"{v['name']} - {v['description']}" for v in MODEL_PRESETS.values()]
        print_menu("模型", options, show_back=True)
        choice = get_choice("请选择", len(options))
        if choice == 0:
            return
        builder.set_model(keys[choice - 1])

        # Step 3: Dataset handler
        print(Style.bold("\n  [步骤 3/5] 选择数据集/因子"))
        keys = list(DATASET_HANDLERS.keys())
        options = [f"{v['name']} - {v['description']}" for v in DATASET_HANDLERS.values()]
        print_menu("数据集", options, show_back=True)
        choice = get_choice("请选择", len(options))
        if choice == 0:
            return
        builder.set_handler(keys[choice - 1])

        # Step 4: Strategy
        print(Style.bold("\n  [步骤 4/5] 选择交易策略"))
        keys = list(STRATEGY_PRESETS.keys())
        options = [f"{v['name']} - {v['description']}" for v in STRATEGY_PRESETS.values()]
        print_menu("策略", options, show_back=True)
        choice = get_choice("请选择", len(options))
        if choice == 0:
            return
        builder.set_strategy(keys[choice - 1])

        # Step 5: Time period
        print(Style.bold("\n  [步骤 5/5] 选择时间段"))
        keys = list(TIME_PRESETS.keys())
        options = [f"{v['name']}" for v in TIME_PRESETS.values()]
        print_menu("时间段", options, show_back=True)
        choice = get_choice("请选择", len(options))
        if choice == 0:
            return
        builder.set_time(keys[choice - 1])

        # Confirm
        print(builder.summary())
        if not confirm("确认以上配置并运行？"):
            return

        self._run_pipeline(builder)

    def template_pipeline(self, builder: ConfigBuilder):
        """使用预设模板"""
        print_section("预设工作流模板")

        templates = [
            {
                "name": "经典价值选股 (LightGBM + Alpha158)",
                "region": "cn", "model": "lightgbm", "handler": "alpha158",
                "strategy": "topk_dropout", "time": "default",
                "desc": "最经典的量化选股策略，使用树模型+158因子",
            },
            {
                "name": "深度学习选股 (LSTM + Alpha158)",
                "region": "cn", "model": "lstm", "handler": "alpha158",
                "strategy": "topk_dropout", "time": "default",
                "desc": "使用LSTM捕获时序依赖关系",
            },
            {
                "name": "保守稳健 (XGBoost + 大范围持仓)",
                "region": "cn", "model": "xgboost", "handler": "alpha158",
                "strategy": "conservative", "time": "default",
                "desc": "持仓分散、换手低的保守策略",
            },
            {
                "name": "Transformer 选股",
                "region": "cn", "model": "transformer", "handler": "alpha158",
                "strategy": "topk_weighted", "time": "default",
                "desc": "使用Transformer自注意力机制选股",
            },
            {
                "name": "快速验证 (线性模型 + 短期)",
                "region": "cn", "model": "linear", "handler": "alpha158",
                "strategy": "topk_dropout", "time": "short",
                "desc": "最快速的端到端验证，适合调试",
            },
        ]

        options = [f"{t['name']}\n       {t['desc']}" for t in templates]
        print_menu("选择模板", options, show_back=True)
        choice = get_choice("请选择", len(templates))
        if choice == 0:
            return

        t = templates[choice - 1]
        builder.set_region(t["region"])
        builder.set_model(t["model"])
        builder.set_handler(t["handler"])
        builder.set_strategy(t["strategy"])
        builder.set_time(t["time"])

        print(builder.summary())
        if not confirm("确认运行此模板？"):
            return

        self._run_pipeline(builder)

    def _run_pipeline(self, builder: ConfigBuilder):
        """执行完整流水线"""
        total_start = time.time()

        print_section("执行量化交易流水线")
        print(Style.bold("  流水线步骤: 数据检查 → 模型训练 → 信号生成 → 策略回测 → 绩效报告"))
        print()

        # Step 1: Check data
        print(Style.info("  ▶ 步骤 1/3: 数据检查"))
        region_conf = REGIONS[builder.region]
        if not check_data_exists(region_conf["provider_uri"]):
            print(Style.warning(f"    未找到 {region_conf['name']} 数据"))
            if confirm("    是否下载数据？"):
                self.data_mgr.download_data(builder.region)
                if not check_data_exists(region_conf["provider_uri"]):
                    print(Style.error("    数据下载失败，流水线终止"))
                    return
            else:
                print(Style.error("    没有数据，流水线终止"))
                return
        print(Style.success("    数据就绪"))

        # Step 2: Train model
        print(Style.info("\n  ▶ 步骤 2/3: 模型训练 + 信号生成"))
        result = self.model_mgr.train_model(
            builder,
            experiment_name=f"pipeline_{builder.model_key}_{builder.handler_key}",
        )
        if result is None:
            print(Style.error("    模型训练失败，流水线终止"))
            return
        model, dataset, recorder = result

        # Step 3: Backtest
        print(Style.info("\n  ▶ 步骤 3/3: 策略回测"))
        backtest_results = self.backtest_eng.run_backtest(
            builder,
            model=model,
            dataset=dataset,
            recorder=recorder,
        )

        # Summary
        total_elapsed = time.time() - total_start
        print_section("流水线完成")
        print(f"  总耗时: {total_elapsed:.1f}s")
        print(f"  实验记录已保存至 MLflow (mlruns/ 目录)")
        print(f"  可通过 mlflow ui 命令查看详细结果")
        print()

    def run_quick(self, builder: ConfigBuilder):
        """非交互式快速运行（用于命令行直接调用）"""
        self._run_pipeline(builder)

"""
模型管理模块
============
负责模型训练、评估和管理。
"""

import time
from typing import Any, Dict, Optional, Tuple

from qlib_trader.config import (
    DATASET_HANDLERS,
    MODEL_PRESETS,
    TIME_PRESETS,
    ConfigBuilder,
)
from qlib_trader.utils import (
    Style,
    confirm,
    get_choice,
    print_menu,
    print_section,
    print_table,
)


class ModelManager:
    """模型管理器"""

    def run_menu(self, builder: ConfigBuilder):
        """模型管理交互菜单"""
        while True:
            print_menu("模型管理", [
                "查看可用模型",
                "选择模型",
                "训练模型",
                "查看已训练的实验",
            ])
            choice = get_choice("请选择", 4)
            if choice == 0:
                return
            elif choice == 1:
                self.show_available_models()
            elif choice == 2:
                self.select_model(builder)
            elif choice == 3:
                self.train_model(builder)
            elif choice == 4:
                self.list_experiments()

    def show_available_models(self):
        """显示所有可用模型"""
        print_section("可用AI模型")

        # Group models by type
        tree_models = []
        dl_models = []
        linear_models = []

        for key, conf in MODEL_PRESETS.items():
            row = [key, conf["name"], conf["description"]]
            if "pytorch" in conf["module_path"] or key in ("lstm", "gru", "transformer", "alstm"):
                dl_models.append(row)
            elif key == "linear":
                linear_models.append(row)
            else:
                tree_models.append(row)

        print(Style.bold("\n  树模型 (训练快, 推荐入门):"))
        print_table(["标识", "名称", "说明"], tree_models, [12, 25, 35])

        if linear_models:
            print(Style.bold("\n  线性模型:"))
            print_table(["标识", "名称", "说明"], linear_models, [12, 25, 35])

        print(Style.bold("\n  深度学习模型 (需要更多计算资源):"))
        print_table(["标识", "名称", "说明"], dl_models, [12, 25, 35])

    def select_model(self, builder: ConfigBuilder):
        """交互式选择模型"""
        print_section("选择模型")
        keys = list(MODEL_PRESETS.keys())
        options = [f"{v['name']} - {v['description']}" for v in MODEL_PRESETS.values()]
        print_menu("可选模型", options, show_back=True)
        choice = get_choice("请选择模型", len(options))
        if choice == 0:
            return
        model_key = keys[choice - 1]
        builder.set_model(model_key)
        print(Style.success(f"\n  已选择模型: {MODEL_PRESETS[model_key]['name']}"))

    def train_model(
        self,
        builder: ConfigBuilder,
        experiment_name: Optional[str] = None,
    ) -> Optional[Tuple[Any, Any, Any]]:
        """训练模型

        Returns:
            (model, dataset, recorder) 或 None（失败时）
        """
        print_section("模型训练")
        model_conf = MODEL_PRESETS[builder.model_key]
        print(f"  模型    : {model_conf['name']}")
        print(f"  数据集  : {DATASET_HANDLERS[builder.handler_key]['name']}")
        print(f"  时间段  : {TIME_PRESETS[builder.time_key]['name']}")
        print()

        if not confirm("确认开始训练？"):
            return None

        if experiment_name is None:
            experiment_name = f"qlib_trader_{builder.model_key}_{builder.handler_key}"

        try:
            import qlib
            from qlib.utils import init_instance_by_config, flatten_dict
            from qlib.workflow import R
            from qlib.workflow.record_temp import SignalRecord, SigAnaRecord

            # Initialize qlib
            init_conf = builder.build_qlib_init_config()
            print(Style.info("\n  [1/4] 初始化 Qlib..."))
            qlib.init(**init_conf)
            print(Style.success("         Qlib 初始化完成"))

            # Build task config
            task_config = builder.build_task_config()

            # Create model and dataset
            print(Style.info("  [2/4] 准备数据集..."))
            model = init_instance_by_config(task_config["model"])
            dataset = init_instance_by_config(task_config["dataset"])
            print(Style.success("         数据集准备完成"))

            # Train
            print(Style.info("  [3/4] 训练模型..."))
            start_time = time.time()

            with R.start(experiment_name=experiment_name):
                R.log_params(**flatten_dict(task_config))
                model.fit(dataset)
                R.save_objects(**{"params.pkl": model})

                elapsed = time.time() - start_time
                print(Style.success(f"         模型训练完成 (耗时 {elapsed:.1f}s)"))

                # Prediction and signal analysis
                print(Style.info("  [4/4] 生成预测信号..."))
                recorder = R.get_recorder()
                sr = SignalRecord(model, dataset, recorder)
                sr.generate()

                sar = SigAnaRecord(recorder)
                sar.generate()
                print(Style.success("         信号分析完成"))

                # Print signal analysis results
                self._print_signal_analysis(recorder)

                return model, dataset, recorder

        except Exception as e:
            print(Style.error(f"\n  训练失败: {e}"))
            import traceback
            traceback.print_exc()
            return None

    def _print_signal_analysis(self, recorder):
        """打印信号分析结果"""
        try:
            sig_ana = recorder.load_object("sig_analysis")
            print_section("信号分析结果")

            if hasattr(sig_ana, "items"):
                for metric_name, value in sig_ana.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric_name}: {value:.6f}")
            elif hasattr(sig_ana, "to_string"):
                print(sig_ana.to_string())
            else:
                print(f"  {sig_ana}")
        except Exception:
            print(Style.info("  (信号分析详情请查看 MLflow 实验记录)"))

    def list_experiments(self):
        """列出已有的实验"""
        print_section("已训练的实验")
        try:
            import qlib
            from qlib.workflow import R

            # Try to init if not already
            try:
                qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn", skip_if_reg=True)
            except Exception:
                pass

            exps = R.list_experiments()
            if not exps:
                print(Style.info("  暂无实验记录"))
                return

            rows = []
            for exp in exps:
                exp_name = exp.name if hasattr(exp, "name") else str(exp)
                exp_id = exp.id if hasattr(exp, "id") else "-"
                rows.append([str(exp_id), exp_name])

            print_table(["ID", "实验名称"], rows, [10, 40])

        except Exception as e:
            print(Style.warning(f"  无法获取实验列表: {e}"))
            print(Style.info("  提示: 需要先初始化 Qlib 并完成至少一次训练"))

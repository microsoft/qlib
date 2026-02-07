"""
配置管理模块
============
管理所有预设配置、用户自定义配置和工作流模板。
"""

import os
import copy
from pathlib import Path
from typing import Any, Dict, Optional

# ============================================================
# 区域 / 市场配置
# ============================================================

REGIONS = {
    "cn": {
        "name": "中国A股",
        "provider_uri": "~/.qlib/qlib_data/cn_data",
        "benchmark": "SH000300",
        "market": "csi300",
        "trade_unit": 100,
        "limit_threshold": 0.095,
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
    },
    "us": {
        "name": "美股",
        "provider_uri": "~/.qlib/qlib_data/us_data",
        "benchmark": "^gspc",
        "market": "sp500",
        "trade_unit": 1,
        "limit_threshold": None,
        "open_cost": 0.0005,
        "close_cost": 0.0005,
        "min_cost": 5,
    },
}

# ============================================================
# 数据集处理器配置
# ============================================================

DATASET_HANDLERS = {
    "alpha158": {
        "name": "Alpha158 (158个Alpha因子)",
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "description": "经典158因子集，包含量价技术指标",
    },
    "alpha360": {
        "name": "Alpha360 (360个Alpha因子)",
        "class": "Alpha360",
        "module_path": "qlib.contrib.data.handler",
        "description": "扩展360因子集，包含日内特征",
    },
}

# ============================================================
# 模型预设配置
# ============================================================

MODEL_PRESETS = {
    "lightgbm": {
        "name": "LightGBM (推荐入门)",
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "description": "微软梯度提升框架，训练快速，效果优秀",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
        },
    },
    "xgboost": {
        "name": "XGBoost",
        "class": "XGBModel",
        "module_path": "qlib.contrib.model.xgboost",
        "description": "极端梯度提升，经典高性能模型",
        "kwargs": {
            "max_depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 2000,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
        },
    },
    "catboost": {
        "name": "CatBoost",
        "class": "CatBoostModel",
        "module_path": "qlib.contrib.model.catboost_model",
        "description": "Yandex提升树模型，自动处理类别特征",
        "kwargs": {
            "loss_function": "RMSE",
            "learning_rate": 0.05,
            "depth": 8,
            "iterations": 2000,
        },
    },
    "linear": {
        "name": "线性模型 (Linear)",
        "class": "LinearModel",
        "module_path": "qlib.contrib.model.linear",
        "description": "简单线性回归，适合基线对比",
        "kwargs": {},
    },
    "lstm": {
        "name": "LSTM (深度学习)",
        "class": "LSTM",
        "module_path": "qlib.contrib.model.pytorch_lstm",
        "description": "长短期记忆网络，捕获时序依赖",
        "kwargs": {
            "d_feat": 158,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.0,
            "n_epochs": 200,
            "lr": 0.001,
            "early_stop": 20,
            "batch_size": 800,
            "metric": "loss",
            "loss": "mse",
            "GPU": 0,
        },
    },
    "gru": {
        "name": "GRU (深度学习)",
        "class": "GRU",
        "module_path": "qlib.contrib.model.pytorch_gru",
        "description": "门控循环单元网络，轻量级时序模型",
        "kwargs": {
            "d_feat": 158,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.0,
            "n_epochs": 200,
            "lr": 0.001,
            "early_stop": 20,
            "batch_size": 800,
            "metric": "loss",
            "loss": "mse",
            "GPU": 0,
        },
    },
    "transformer": {
        "name": "Transformer (深度学习)",
        "class": "Transformer",
        "module_path": "qlib.contrib.model.pytorch_transformer",
        "description": "Transformer模型，自注意力机制",
        "kwargs": {
            "d_feat": 158,
            "d_model": 64,
            "nhead": 2,
            "num_layers": 2,
            "dropout": 0.0,
            "n_epochs": 200,
            "lr": 0.0001,
            "early_stop": 20,
            "batch_size": 800,
            "metric": "loss",
            "loss": "mse",
            "GPU": 0,
        },
    },
    "alstm": {
        "name": "ALSTM (注意力LSTM)",
        "class": "ALSTM",
        "module_path": "qlib.contrib.model.pytorch_alstm",
        "description": "带注意力机制的LSTM模型",
        "kwargs": {
            "d_feat": 158,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.0,
            "n_epochs": 200,
            "lr": 0.001,
            "early_stop": 20,
            "batch_size": 800,
            "metric": "loss",
            "loss": "mse",
            "rnn_type": "GRU",
            "GPU": 0,
        },
    },
}

# ============================================================
# 策略预设配置
# ============================================================

STRATEGY_PRESETS = {
    "topk_dropout": {
        "name": "TopK-Dropout 策略 (推荐)",
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "description": "持有预测值最高的K只股票，定期淘汰表现差的",
        "kwargs": {
            "topk": 50,
            "n_drop": 5,
        },
    },
    "topk_weighted": {
        "name": "TopK 加权策略",
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "description": "较少持仓集中度，大幅换手",
        "kwargs": {
            "topk": 30,
            "n_drop": 10,
        },
    },
    "conservative": {
        "name": "保守策略",
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "description": "持仓更多、换手更低的保守策略",
        "kwargs": {
            "topk": 80,
            "n_drop": 3,
        },
    },
}

# ============================================================
# 时间段预设
# ============================================================

TIME_PRESETS = {
    "default": {
        "name": "标准时段 (2008-2020)",
        "train": ("2008-01-01", "2014-12-31"),
        "valid": ("2015-01-01", "2016-12-31"),
        "test": ("2017-01-01", "2020-08-01"),
        "data_start": "2008-01-01",
        "data_end": "2020-08-01",
    },
    "recent": {
        "name": "近期时段 (2015-2023)",
        "train": ("2015-01-01", "2019-12-31"),
        "valid": ("2020-01-01", "2020-12-31"),
        "test": ("2021-01-01", "2023-12-31"),
        "data_start": "2015-01-01",
        "data_end": "2023-12-31",
    },
    "short": {
        "name": "短期快速验证 (2016-2020)",
        "train": ("2016-01-01", "2017-12-31"),
        "valid": ("2018-01-01", "2018-12-31"),
        "test": ("2019-01-01", "2020-08-01"),
        "data_start": "2016-01-01",
        "data_end": "2020-08-01",
    },
    "long": {
        "name": "长周期 (2005-2023)",
        "train": ("2005-01-01", "2015-12-31"),
        "valid": ("2016-01-01", "2017-12-31"),
        "test": ("2018-01-01", "2023-12-31"),
        "data_start": "2005-01-01",
        "data_end": "2023-12-31",
    },
}

# ============================================================
# 回测配置
# ============================================================

BACKTEST_DEFAULTS = {
    "account": 100_000_000,   # 初始资金 1亿
    "freq": "day",
    "deal_price": "close",
}


# ============================================================
# 配置构建器
# ============================================================

class ConfigBuilder:
    """构建完整的 qlib 工作流配置"""

    def __init__(self):
        self.region: str = "cn"
        self.model_key: str = "lightgbm"
        self.handler_key: str = "alpha158"
        self.strategy_key: str = "topk_dropout"
        self.time_key: str = "default"
        self.account: int = BACKTEST_DEFAULTS["account"]
        self.custom_overrides: Dict[str, Any] = {}

    def set_region(self, region: str) -> "ConfigBuilder":
        if region not in REGIONS:
            raise ValueError(f"不支持的区域: {region}，可选: {list(REGIONS.keys())}")
        self.region = region
        return self

    def set_model(self, model_key: str) -> "ConfigBuilder":
        if model_key not in MODEL_PRESETS:
            raise ValueError(f"不支持的模型: {model_key}，可选: {list(MODEL_PRESETS.keys())}")
        self.model_key = model_key
        return self

    def set_handler(self, handler_key: str) -> "ConfigBuilder":
        if handler_key not in DATASET_HANDLERS:
            raise ValueError(f"不支持的数据集: {handler_key}，可选: {list(DATASET_HANDLERS.keys())}")
        self.handler_key = handler_key
        return self

    def set_strategy(self, strategy_key: str) -> "ConfigBuilder":
        if strategy_key not in STRATEGY_PRESETS:
            raise ValueError(f"不支持的策略: {strategy_key}，可选: {list(STRATEGY_PRESETS.keys())}")
        self.strategy_key = strategy_key
        return self

    def set_time(self, time_key: str) -> "ConfigBuilder":
        if time_key not in TIME_PRESETS:
            raise ValueError(f"不支持的时间段: {time_key}，可选: {list(TIME_PRESETS.keys())}")
        self.time_key = time_key
        return self

    def set_account(self, account: int) -> "ConfigBuilder":
        self.account = account
        return self

    def build_qlib_init_config(self) -> Dict[str, Any]:
        """构建 qlib.init() 参数"""
        region_conf = REGIONS[self.region]
        return {
            "provider_uri": region_conf["provider_uri"],
            "region": self.region,
        }

    def build_task_config(self) -> Dict[str, Any]:
        """构建模型训练任务配置"""
        model_conf = MODEL_PRESETS[self.model_key]
        handler_conf = DATASET_HANDLERS[self.handler_key]
        time_conf = TIME_PRESETS[self.time_key]
        region_conf = REGIONS[self.region]

        task = {
            "model": {
                "class": model_conf["class"],
                "module_path": model_conf["module_path"],
                "kwargs": copy.deepcopy(model_conf["kwargs"]),
            },
            "dataset": {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": handler_conf["class"],
                        "module_path": handler_conf["module_path"],
                        "kwargs": {
                            "start_time": time_conf["data_start"],
                            "end_time": time_conf["data_end"],
                            "fit_start_time": time_conf["train"][0],
                            "fit_end_time": time_conf["train"][1],
                            "instruments": region_conf["market"],
                        },
                    },
                    "segments": {
                        "train": time_conf["train"],
                        "valid": time_conf["valid"],
                        "test": time_conf["test"],
                    },
                },
            },
        }
        return task

    def build_backtest_config(self) -> Dict[str, Any]:
        """构建回测配置"""
        region_conf = REGIONS[self.region]
        time_conf = TIME_PRESETS[self.time_key]
        strategy_conf = STRATEGY_PRESETS[self.strategy_key]

        return {
            "executor": {
                "class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {
                    "time_per_step": "day",
                    "generate_portfolio_metrics": True,
                },
            },
            "strategy": {
                "class": strategy_conf["class"],
                "module_path": strategy_conf["module_path"],
                "kwargs": copy.deepcopy(strategy_conf["kwargs"]),
            },
            "backtest": {
                "start_time": time_conf["test"][0],
                "end_time": time_conf["test"][1],
                "account": self.account,
                "benchmark": region_conf["benchmark"],
                "exchange_kwargs": {
                    "freq": BACKTEST_DEFAULTS["freq"],
                    "limit_threshold": region_conf["limit_threshold"],
                    "deal_price": BACKTEST_DEFAULTS["deal_price"],
                    "open_cost": region_conf["open_cost"],
                    "close_cost": region_conf["close_cost"],
                    "min_cost": region_conf["min_cost"],
                },
            },
        }

    def summary(self) -> str:
        """返回当前配置摘要"""
        region_conf = REGIONS[self.region]
        model_conf = MODEL_PRESETS[self.model_key]
        handler_conf = DATASET_HANDLERS[self.handler_key]
        strategy_conf = STRATEGY_PRESETS[self.strategy_key]
        time_conf = TIME_PRESETS[self.time_key]

        lines = [
            "=" * 60,
            "  当前配置摘要",
            "=" * 60,
            f"  市场区域  : {region_conf['name']} ({self.region})",
            f"  基准指数  : {region_conf['benchmark']}",
            f"  选股范围  : {region_conf['market']}",
            f"  AI 模型   : {model_conf['name']}",
            f"  数据集    : {handler_conf['name']}",
            f"  交易策略  : {strategy_conf['name']}",
            f"  时间范围  : {time_conf['name']}",
            f"    训练集  : {time_conf['train'][0]} ~ {time_conf['train'][1]}",
            f"    验证集  : {time_conf['valid'][0]} ~ {time_conf['valid'][1]}",
            f"    测试集  : {time_conf['test'][0]} ~ {time_conf['test'][1]}",
            f"  初始资金  : {self.account:,.0f}",
            "=" * 60,
        ]
        return "\n".join(lines)

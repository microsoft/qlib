#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
训练和回测脚本

参考 examples/workflow_by_code.py，从config.yaml读取配置

使用方法:
    python train.py train --config config.yaml
    python train.py plot --recorder_id <recorder_id>
"""
import sys
from pathlib import Path
from datetime import datetime

import yaml
import fire
import qlib
from loguru import logger
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入绘图模块
from plot_utils import (
    get_plot_output_dir,
    generate_all_plots,
    print_statistics,
    save_statistics_to_file,
)


def load_config(config_path):
    """加载配置文件"""
    config_path = Path(config_path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def get_data_handler_config(data_config, train_config):
    """获取数据处理器配置"""
    # 使用数据配置中的日期范围（整体数据范围），而非测试集时间段
    # 转换日期格式：20080101 -> 2008-01-01
    start_date = str(data_config['start_date'])
    end_date = str(data_config['end_date'])

    return {
        "start_time": f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}",
        "end_time": f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}",
        "fit_start_time": train_config['segments']['train'][0],
        "fit_end_time": train_config['segments']['train'][1],
        "instruments": train_config['instruments'],
    }


def get_model_config(train_config):
    """
    获取模型配置

    支持多种模型，通过 train_config 中的 model 字段指定:
    - LightGBM: qlib.contrib.model.gbdt (LGBModel)
    - XGBoost: qlib.contrib.model.xgboost (XGBModel)
    - Linear: qlib.contrib.model.linear (LinearModel)
    - CatBoost: qlib.contrib.model.catboost (CatBoostModel)

    Parameters
    ----------
    train_config : dict
        训练配置，包含 model 字段

    Returns
    -------
    dict
        模型配置字典
    """
    model_name = train_config.get('model', 'LightGBM')

    # 基础模型配置模板
    model_configs = {
        'LightGBM': {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.02,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 20,
            },
        },
        'XGBoost': {
            "class": "XGBModel",
            "module_path": "qlib.contrib.model.xgboost",
            "kwargs": {
                "eval_metric": "rmse",
                "colsample_bytree": 0.8879,
                "eta": 0.0421,
                "max_depth": 8,
                "n_estimators": 647,
                "subsample": 0.8789,
                "nthread": 20,
            },
        },
        'Linear': {
            "class": "LinearModel",
            "module_path": "qlib.contrib.model.linear",
            "kwargs": {
                "reg": 0.0001,
            },
        },
        'CatBoost': {
            "class": "CatBoostModel",
            "module_path": "qlib.contrib.model.catboost",
            "kwargs": {
                "loss_function": "RMSE",
                "iterations": 500,
                "learning_rate": 0.03,
                "depth": 6,
                "l2_leaf_reg": 3,
                "random_seed": 42,
                "thread_count": 20,
            },
        },
        'LSTM': {
            "class": "LSTM",
            "module_path": "qlib.contrib.model.pytorch_lstm_ts",
            "kwargs": {
                "d_feat": 20,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.0,
                "n_epochs": 200,
                "lr": 1e-3,
                "early_stop": 10,
                "batch_size": 800,
                "metric": "loss",
                "loss": "mse",
                "n_jobs": 20,
                "GPU": 0,
            },
        },
        'GRU': {
            "class": "GRU",
            "module_path": "qlib.contrib.model.pytorch_gru",
            "kwargs": {
                "d_feat": 20,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.0,
                "n_epochs": 200,
                "lr": 1e-3,
                "early_stop": 10,
                "batch_size": 800,
                "metric": "loss",
                "loss": "mse",
                "n_jobs": 20,
                "GPU": 0,
            },
        },
    }

    # 检查是否有自定义模型配置
    if 'model_config' in train_config:
        # 使用配置文件中的自定义模型配置
        return train_config['model_config']

    # 使用预设模板，支持大小写不敏感匹配
    model_key = next((k for k in model_configs.keys() if k.lower() == model_name.lower()), 'LightGBM')
    return model_configs[model_key]


def get_dataset_config(handler_config, train_config):
    """获取数据集配置"""
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": handler_config,
            },
            "segments": {
                "train": train_config['segments']['train'],
                "valid": train_config['segments']['valid'],
                "test": train_config['segments']['test'],
            },
        },
    }

    # 对于时序模型(LSTM, GRU等)，使用 TSDatasetH
    model_name = train_config.get('model', '').lower()
    if model_name in ['lstm', 'gru']:
        dataset_config["class"] = "TSDatasetH"
        dataset_config["kwargs"]["step_len"] = train_config.get('step_len', 20)

    return dataset_config


def get_port_analysis_config(backtest_config, model, dataset):
    """获取回测配置"""
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
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": backtest_config['topk'],
                "n_drop": backtest_config['n_drop'],
            },
        },
        "backtest": {
            "start_time": backtest_config['start_time'],
            "end_time": backtest_config['end_time'],
            "account": backtest_config['account'],
            "benchmark": backtest_config['benchmark'],
            "exchange_kwargs": backtest_config.get('exchange_kwargs', {
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            }),
        },
    }


def train(config_path=None):
    """
    训练模型并回测

    Parameters
    ----------
    config_path : str
        配置文件路径，默认为 train.py 同目录下的 config.yaml
    """
    # 默认使用脚本同目录下的 config.yaml
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)
    # 加载配置
    config = load_config(config_path)
    data_config = config['data']
    train_config = config['train']
    backtest_config = config['backtest']

    # 初始化qlib
    qlib_dir = Path(data_config['qlib_dir']).expanduser()
    provider_uri = str(qlib_dir)

    logger.info(f"初始化qlib，数据目录: {provider_uri}")
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # 构建配置
    handler_config = get_data_handler_config(data_config, train_config)
    dataset_config = get_dataset_config(handler_config, train_config)
    model_config = get_model_config(train_config)

    # 初始化模型和数据集
    logger.info("初始化模型...")
    model = init_instance_by_config(model_config)

    logger.info("初始化数据集...")
    dataset = init_instance_by_config(dataset_config)

    # # 显示数据集信息
    # train_data = dataset.prepare("train")
    # logger.info(f"训练集形状: {train_data.shape}")
    # logger.info(f"训练集列数: {len(train_data.columns)}")

    # 获取回测配置
    port_analysis_config = get_port_analysis_config(backtest_config, model, dataset)

    # 开始实验
    experiment_name = f"my-quant-{train_config['model']}"
    logger.info(f"开始实验: {experiment_name}")

    with R.start(experiment_name=experiment_name):
        # 保存配置
        R.log_params(
            model=model_config['class'],
            provider_uri=provider_uri,
            instruments=train_config['instruments'],
            train_period=train_config['segments']['train'],
            valid_period=train_config['segments']['valid'],
            test_period=train_config['segments']['test'],
        )

        # 训练模型
        logger.info("开始训练模型...")
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        # 生成预测
        logger.info("生成预测信号...")
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # 信号分析
        logger.info("信号分析...")
        sar = SigAnaRecord(recorder)
        sar.generate()

        # 回测
        logger.info("开始回测...")
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()

        # 生成图表
        logger.info("生成分析图表...")
        plots_dir = get_plot_output_dir(Path(__file__).parent, experiment_name)
        results = generate_all_plots(recorder, plots_dir, experiment_name)

        # 打印统计信息
        print_statistics(results['statistics'])

        # 保存统计信息到文件
        stats_file = plots_dir / "statistics.txt"
        save_statistics_to_file(results['statistics'], str(stats_file))
        logger.info(f"统计信息已保存到: {stats_file}")

        logger.info(f"图表已保存到: {plots_dir}")

    logger.info("训练和回测完成!")
    logger.info(f"实验结果保存在: {qlib_dir.parent / 'mlruns'}")


def dump_bin(config_path=None):
    """
    将CSV数据转换为qlib的bin格式

    Parameters
    ----------
    config_path : str
        配置文件路径，默认为 train.py 同目录下的 config.yaml
    """
    import subprocess

    # 默认使用脚本同目录下的 config.yaml
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    config = load_config(config_path)
    data_config = config['data']

    # # 调用dump_bin脚本
    # cmd = [
    #     sys.executable,
    #     str(Path(__file__).parent.parent.parent / "scripts" / "dump_bin.py"),
    #     "dump_all",
    #     "--data_path", data_config['normalize_dir'],
    #     "--qlib_dir", data_config['qlib_dir'],
    #     "--freq", "day",
    # ]
    # 调用dump_bin脚本
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent.parent / "scripts" / "dump_bin.py"),
        "dump_all",
        "--data_path", data_config['normalize_dir'], 
        "--qlib_dir", data_config['qlib_dir'],
        "--freq", "day",
        # 保留这两个配置（解决之前的字符串报错）
        "--date_field_name", "date",     
        "--exclude_fields", "symbol,date", 
    ]

    logger.info(f"执行: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    logger.info("数据转换完成!")


def plot(recorder_id: str = None, experiment_name: str = None, config_path: str = None):
    """
    为已完成的实验生成分析图表

    Parameters
    ----------
    recorder_id : str
        实验recorder ID (如 20400880540b4ae184e342a6f6659af6)
    experiment_name : str
        实验名称 (如 my-quant-LightGBM)
        如果不指定recorder_id，将使用最新的实验
    config_path : str
        配置文件路径，用于初始化qlib
    """
    from qlib.workflow import R

    # 初始化qlib (需要从配置文件获取数据路径)
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    config = load_config(config_path)
    data_config = config['data']
    qlib_dir = Path(data_config['qlib_dir']).expanduser()

    logger.info(f"初始化qlib，数据目录: {qlib_dir}")
    qlib.init(provider_uri=str(qlib_dir), region=REG_CN)

    # 确定要使用的recorder
    if recorder_id is None and experiment_name is None:
        # 使用最新的实验
        experiment_name = "my-quant-LightGBM"

    if recorder_id is not None:
        recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment_name)
    else:
        # 获取实验下最新的recorder
        try:
            experiment = R.get_exp(experiment_name=experiment_name, create=False)
            recorders = list(experiment.list_recorders(rtype="list"))
            if not recorders:
                raise ValueError(f"实验 {experiment_name} 没有找到recorder")
            recorder = recorders[-1]  # 使用最新的recorder
            recorder_id = recorder.id
            logger.info(f"使用最新的recorder: {recorder_id}")
        except Exception as e:
            logger.error(f"获取recorder失败: {e}")
            return

    # 获取实验名称
    if experiment_name is None:
        experiment_name = recorder.experiment_name

    logger.info(f"为实验 {experiment_name} (recorder: {recorder_id}) 生成图表...")

    # 创建输出目录
    plots_dir = get_plot_output_dir(Path(__file__).parent, experiment_name)

    # 生成图表
    results = generate_all_plots(recorder, plots_dir, experiment_name)

    # 打印统计信息
    print_statistics(results['statistics'])

    # 保存统计信息到文件
    stats_file = plots_dir / "statistics.txt"
    save_statistics_to_file(results['statistics'], str(stats_file))
    logger.info(f"统计信息已保存到: {stats_file}")

    logger.info(f"图表已保存到: {plots_dir}")
    logger.info(f"生成的文件: {list(results['saved_files'].keys())}")


def create_config(model: str = "LightGBM", config_path: str = None):
    """
    生成指定模型的配置文件

    Parameters
    ----------
    model : str
        模型名称: LightGBM, XGBoost, Linear, CatBoost, LSTM, GRU
    config_path : str
        配置文件路径
    """
    base_config = load_config(Path(__file__).parent / "config.yaml")
    base_config['train']['model'] = model

    # 移除自定义配置，使用默认模板
    if 'model_config' in base_config['train']:
        del base_config['train']['model_config']

    if config_path is None:
        config_path = Path(__file__).parent / f"config_{model.lower()}.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(base_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    logger.info(f"配置文件已生成: {config_path}")
    logger.info(f"使用模型: {model}")
    logger.info("\n使用方法:")
    logger.info(f"  python train.py train --config {config_path}")


def main():
    """主函数"""
    fire.Fire({
        'train': train,
        'dump_bin': dump_bin,
        'plot': plot,
        'create_config': create_config,
    })


if __name__ == "__main__":
    main()

#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
Long-Short workflow by code (Crypto Perp).

This script mirrors `workflow_by_code_longshort.py` but switches to a crypto futures
dataset/provider and sets the benchmark to BTCUSDT. Other parts are kept the same.
"""
import os
import importlib.util
from pathlib import Path
import plotly.io as pio
import qlib
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord


if __name__ == "__main__":
    # Initialize with crypto perp data provider (ensure this path exists in your env)
    provider_uri = "~/.qlib/qlib_data/crypto_data_perp"
    qlib.init(provider_uri=provider_uri)

    # Dataset & model
    data_handler_config = {
        "start_time": "2019-01-02",
        "end_time": "2025-08-07",
        "fit_start_time": "2019-01-02",
        "fit_end_time": "2022-12-19",
        "instruments": "all",
        "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
    }

    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments": {
                # train uses fit window; split the rest to valid/test roughly
                "train": (data_handler_config["fit_start_time"], data_handler_config["fit_end_time"]),
                "valid": ("2022-12-20", "2023-12-31"),
                "test": ("2024-01-01", data_handler_config["end_time"]),
            },
        },
    }

    model_config = {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
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
    }

    model = init_instance_by_config(model_config)
    dataset = init_instance_by_config(dataset_config)

    # Load CryptoPortAnaRecord from crypto-qlib/crypto_qlib_config.py
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
    crypto_cfg_path = os.path.join(project_root, "crypto-qlib", "crypto_qlib_config.py")
    spec = importlib.util.spec_from_file_location("crypto_qlib_config", crypto_cfg_path)
    crypto_cfg = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(crypto_cfg)
    CryptoPortAnaRecord = crypto_cfg.CryptoPortAnaRecord

    # Align backtest time to test segment
    test_start, test_end = dataset_config["kwargs"]["segments"]["test"]

    port_analysis_config = {
        "executor": {
            "class": "ShortableExecutor",
            "module_path": "qlib.backtest.shortable_backtest",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "LongShortTopKStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk_long": 20,
                "topk_short": 20,
                "n_drop_long": 10,
                "n_drop_short": 10,
                "hold_thresh": 3,
                "only_tradable": True,
                "forbid_all_trade_at_limit": False,
            },
        },
        "backtest": {
            "start_time": test_start,
            "end_time": test_end,
            "account": 100000000,
            "benchmark": "BTCUSDT",
            "exchange_kwargs": {
                "exchange": {
                    "class": "ShortableExchange",
                    "module_path": "qlib.backtest.shortable_exchange",
                },
                "freq": "day",
                # Crypto has no daily price limit; set to 0.0 to avoid false limit locks
                "limit_threshold": 0.0,
                "deal_price": "close",
                "open_cost": 0.0002,
                "close_cost": 0.0005,
                "min_cost": 0,
            },
        },
    }

    # Preview prepared data
    example_df = dataset.prepare("train")
    print(example_df.head())

    # Start experiment
    with R.start(experiment_name="workflow_longshort_crypto"):
        R.log_params(**flatten_dict({"model": model_config, "dataset": dataset_config}))
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        # Prediction
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # Signal Analysis
        sar = SigAnaRecord(recorder)
        sar.generate()

        # Backtest with long-short strategy (Crypto metrics)
        par = CryptoPortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()

        # Visualization (save figures like workflow_by_code.ipynb)
        from qlib.contrib.report.analysis_position import report as qreport
        from qlib.contrib.report.analysis_position import risk_analysis as qrisk

        report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

        figs_dir = Path(recorder.artifact_uri).joinpath("portfolio_analysis/figs").resolve()
        os.makedirs(figs_dir, exist_ok=True)

        # Portfolio report graphs
        rep_figs = qreport.report_graph(report_df, show_notebook=False)
        for idx, fig in enumerate(rep_figs, start=1):
            pio.write_html(fig, str(figs_dir / f"report_graph_{idx}.html"), auto_open=False, include_plotlyjs="cdn")

        # Risk analysis graphs
        risk_figs = qrisk.risk_analysis_graph(analysis_df, report_df, show_notebook=False)
        for idx, fig in enumerate(risk_figs, start=1):
            pio.write_html(fig, str(figs_dir / f"risk_graph_{idx}.html"), auto_open=False, include_plotlyjs="cdn")

        print(f"Saved figures to: {figs_dir}")



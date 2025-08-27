#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
Long-Short workflow by code (Crypto Perp).

This script mirrors `workflow_by_code_longshort.py` but switches to a crypto futures
dataset/provider and sets the benchmark to BTCUSDT. Other parts are kept the same.
"""
# pylint: disable=C0301

import sys
import multiprocessing as mp
import os
import qlib
from qlib.utils import init_instance_by_config, flatten_dict


if __name__ == "__main__":
    # Windows compatibility: spawn mode needs freeze_support and avoid heavy top-level imports
    if sys.platform.startswith("win"):
        mp.freeze_support()
    # Emulate Windows spawn on POSIX if needed
    if os.environ.get("WINDOWS_SPAWN_TEST") == "1" and not sys.platform.startswith("win"):
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
    # Lazy imports to avoid circular import issues on Windows spawn mode
    from qlib.workflow import R
    from qlib.workflow.record_temp import SignalRecord, SigAnaRecord
    from qlib.data import D

    # Initialize with crypto perp data provider (ensure this path exists in your env)
    PROVIDER_URI = "~/.qlib/qlib_data/crypto_data_perp"
    qlib.init(provider_uri=PROVIDER_URI, kernels=1)

    # Auto-select benchmark by data source: cn_data -> SH000300; crypto -> BTCUSDT
    # Fallback: if path not resolvable, default to SH000300 for safety
    try:
        from qlib.config import C

        data_roots = {k: str(C.dpm.get_data_uri(k)) for k in C.dpm.provider_uri.keys()}
        DATA_ROOTS_STR = " ".join(data_roots.values()).lower()
        IS_CN = ("cn_data" in DATA_ROOTS_STR) or ("cn\x5fdata" in DATA_ROOTS_STR)
        BENCHMARK_AUTO = "SH000300" if IS_CN else "BTCUSDT"
    except Exception:  # pylint: disable=W0718
        BENCHMARK_AUTO = "SH000300"

    # Dataset & model
    data_handler_config = {
        "start_time": "2019-01-02",
        "end_time": "2025-08-07",
        "fit_start_time": "2019-01-02",
        "fit_end_time": "2022-12-19",
        "instruments": "all",
        "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
    }

    DEBUG_FAST = os.environ.get("FAST_DEBUG") == "1"
    if DEBUG_FAST:
        # Use the latest available calendar to auto-derive a tiny, non-empty window
        cal = D.calendar(freq="day", future=False)
        if len(cal) >= 45:
            end_dt = cal[-1]
            # last 45 days: 20d fit, 10d valid, 15d test
            fit_start_dt = cal[-45]
            fit_end_dt = cal[-25]
            valid_start_dt = cal[-24]
            valid_end_dt = cal[-15]
            test_start_dt = cal[-14]
            test_end_dt = end_dt
            data_handler_config.update(
                {
                    "fit_start_time": fit_start_dt,
                    "fit_end_time": fit_end_dt,
                    "start_time": fit_start_dt,
                    "end_time": end_dt,
                }
            )

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

    # Predefine debug dates to avoid linter used-before-assignment warning
    VALID_START_DT = VALID_END_DT = TEST_START_DT = TEST_END_DT = None

    if DEBUG_FAST and len(D.calendar(freq="day", future=False)) >= 45:
        dataset_config["kwargs"]["segments"] = {
            "train": (data_handler_config["fit_start_time"], data_handler_config["fit_end_time"]),
            "valid": (VALID_START_DT, VALID_END_DT),
            "test": (TEST_START_DT, TEST_END_DT),
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

    if DEBUG_FAST:
        model_config["kwargs"].update({"num_threads": 2, "num_boost_round": 10})

    model = init_instance_by_config(model_config)
    dataset = init_instance_by_config(dataset_config)

    # Prefer contrib's crypto version; fallback to default PortAnaRecord (no external local dependency)
    try:
        from qlib.contrib.workflow.crypto_record_temp import CryptoPortAnaRecord as PortAnaRecord  # type: ignore

        print("Using contrib's crypto version of CryptoPortAnaRecord as PortAnaRecord")
    except Exception:  # pylint: disable=W0718
        from qlib.workflow.record_temp import PortAnaRecord

        print("Using default version of PortAnaRecord")

    # Align backtest time to test segment
    test_start, test_end = dataset_config["kwargs"]["segments"]["test"]

    # Strategy params (shrink for fast validation)
    TOPK_L, TOPK_S, DROP_L, DROP_S = 20, 20, 10, 10
    if DEBUG_FAST:
        TOPK_L = TOPK_S = 5
        DROP_L = DROP_S = 1

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
                "topk_long": TOPK_L,
                "topk_short": TOPK_S,
                "n_drop_long": DROP_L,
                "n_drop_short": DROP_S,
                "hold_thresh": 3,
                "only_tradable": True,
                "forbid_all_trade_at_limit": False,
            },
        },
        "backtest": {
            "start_time": test_start,
            "end_time": test_end,
            "account": 100000000,
            "benchmark": BENCHMARK_AUTO,
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
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import qlib
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config

from rqalpha_lgbm_param_sweep import add_qlib_runtime_args, build_qlib_init_kwargs
from rqalpha_lgbm_preset_backtest import (
    apply_candidate_filters,
    apply_signal_filters,
    build_benchmark,
    shift_signal_dates,
    summarize_report,
)


def ensure_torch_installed():
    if importlib.util.find_spec("torch") is None:
        raise SystemExit(
            "PyTorch is not installed in the current Python environment. "
            "Run `python -m pip install torch --index-url https://download.pytorch.org/whl/cu121` first."
        )


def sanitize_ts_batch(data, clip_value: float):
    import torch

    data = data.clone()
    feature = data[:, :, 0:-1]
    label = data[:, -1, -1]

    feature = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
    if clip_value > 0:
        feature = torch.clamp(feature, min=-clip_value, max=clip_value)

    label = torch.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0)
    if clip_value > 0:
        label = torch.clamp(label, min=-clip_value, max=clip_value)
    return feature, label


def build_ts_dataset_config(
    instruments: str,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    test_end: str,
    step_len: int,
):
    return {
        "class": "TSDatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "instruments": instruments,
                    "start_time": train_start,
                    "end_time": test_end,
                    "fit_start_time": train_start,
                    "fit_end_time": train_end,
                },
            },
            "segments": {
                "train": (train_start, train_end),
                "valid": (valid_start, valid_end),
                "test": (test_start, test_end),
            },
            "step_len": step_len,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run a GRU/LSTM time-series stock-selection baseline on RQAlpha-converted Qlib data.")
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--train-start", default="2010-01-01")
    parser.add_argument("--train-end", default="2022-12-31")
    parser.add_argument("--valid-start", default="2023-01-01")
    parser.add_argument("--valid-end", default="2023-12-31")
    parser.add_argument("--test-start", default="2024-01-01")
    parser.add_argument("--test-end", default="2024-12-31")
    add_qlib_runtime_args(parser, default_device="gpu")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model-type", choices=("gru", "lstm"), default="gru")
    parser.add_argument("--instruments", default="all")
    parser.add_argument("--topk", type=int, default=80)
    parser.add_argument("--n-drop", type=int, default=10)
    parser.add_argument("--min-history-days", type=int, default=252)
    parser.add_argument("--signal-shift-days", type=int, default=1)
    parser.add_argument("--deal-price", default="close")
    parser.add_argument("--benchmark", default="")
    parser.add_argument("--min-price", type=float, default=0.0)
    parser.add_argument("--min-adv", type=float, default=0.0)
    parser.add_argument("--max-volatility", type=float, default=0.0)
    parser.add_argument("--liquidity-lookback", type=int, default=20)
    parser.add_argument("--volatility-lookback", type=int, default=20)
    parser.add_argument("--account", type=float, default=10000000)
    parser.add_argument("--step-len", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--early-stop", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--n-jobs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-save-path", default="artifacts/rqalpha_gru_ts_best.pt")
    args = parser.parse_args()

    ensure_torch_installed()
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    provider_uri = str(Path(args.provider_uri).expanduser().resolve())
    wall_start = time.perf_counter()
    print(f"provider_uri={provider_uri}", flush=True)
    print(f"device={args.device}", flush=True)
    print(f"dataset_cache={args.dataset_cache}", flush=True)
    print(f"gpu_available={torch.cuda.is_available()}", flush=True)
    if args.dataset_cache == "simple":
        print(f"local_cache_path={Path(args.local_cache_path).expanduser().resolve()}", flush=True)

    qlib.init(**build_qlib_init_kwargs(args, provider_uri))
    print("qlib init done", flush=True)

    t0 = time.perf_counter()
    dataset = init_instance_by_config(
        build_ts_dataset_config(
            args.instruments,
            args.train_start,
            args.train_end,
            args.valid_start,
            args.valid_end,
            args.test_start,
            args.test_end,
            args.step_len,
        )
    )
    print(f"ts dataset init done, elapsed={time.perf_counter() - t0:.2f}s", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.device == "gpu" else "cpu")

    train_ds = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    valid_ds = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    test_ds = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
    train_ds.config(fillna_type="ffill+bfill")
    valid_ds.config(fillna_type="ffill+bfill")
    test_ds.config(fillna_type="ffill+bfill")

    rnn_class = nn.GRU if args.model_type == "gru" else nn.LSTM
    model = rnn_class(
        input_size=158,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        batch_first=True,
        dropout=args.dropout if args.num_layers > 1 else 0.0,
    ).to(device)
    fc_out = nn.Linear(args.hidden_size, 1).to(device)
    params = list(model.parameters()) + list(fc_out.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    model_save_path = Path(args.model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(args.gpu)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_jobs, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_jobs, drop_last=False)

    t_fit = time.perf_counter()
    best_valid_loss = float("inf")
    best_epoch = -1
    stop_rounds = 0
    for epoch in range(args.n_epochs):
        model.train()
        fc_out.train()
        train_losses = []
        for data in train_loader:
            feature, label = sanitize_ts_batch(data, clip_value=10.0)
            feature = feature.to(device).float()
            label = label.to(device).float()
            output, _ = model(feature)
            pred = fc_out(output[:, -1, :]).squeeze(-1)
            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 3.0)
            optimizer.step()
            if torch.isfinite(loss):
                train_losses.append(float(loss.item()))

        model.eval()
        fc_out.eval()
        valid_losses = []
        with torch.no_grad():
            for data in valid_loader:
                feature, label = sanitize_ts_batch(data, clip_value=10.0)
                feature = feature.to(device).float()
                label = label.to(device).float()
                output, _ = model(feature)
                pred = fc_out(output[:, -1, :]).squeeze(-1)
                loss = loss_fn(pred, label)
                if torch.isfinite(loss):
                    valid_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        valid_loss = float(np.mean(valid_losses)) if valid_losses else float("inf")
        print(f"{args.model_type}_ts epoch={epoch + 1}, train_loss={train_loss:.6f}, valid_loss={valid_loss:.6f}", flush=True)

        if np.isfinite(valid_loss) and valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch + 1
            stop_rounds = 0
            torch.save({"rnn": model.state_dict(), "fc_out": fc_out.state_dict()}, model_save_path)
        else:
            stop_rounds += 1
            if stop_rounds >= args.early_stop:
                print(f"{args.model_type}_ts early stop at epoch={epoch + 1}", flush=True)
                break

    checkpoint = torch.load(model_save_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["rnn"])
    fc_out.load_state_dict(checkpoint["fc_out"])
    fit_elapsed = time.perf_counter() - t_fit
    print(
        f"{args.model_type}_ts fit done, elapsed={fit_elapsed:.2f}s, best_epoch={best_epoch}, best_valid_loss={best_valid_loss:.6f}",
        flush=True,
    )

    if torch.cuda.is_available():
        print(
            f"torch_cuda_max_memory_allocated_gb={torch.cuda.max_memory_allocated(args.gpu) / (1024**3):.2f}",
            flush=True,
        )

    t_pred = time.perf_counter()
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_jobs, drop_last=False)
    preds = []
    model.eval()
    fc_out.eval()
    with torch.no_grad():
        for data in test_loader:
            feature, _ = sanitize_ts_batch(data, clip_value=10.0)
            feature = feature.to(device).float()
            output, _ = model(feature)
            pred = fc_out(output[:, -1, :]).squeeze(-1).detach().cpu().numpy()
            preds.append(pred)
    pred = pd.Series(np.concatenate(preds), index=test_ds.get_index())
    pred = apply_signal_filters(
        pred,
        args.instruments,
        args.train_start,
        args.test_end,
        args.min_history_days,
    )
    pred = apply_candidate_filters(
        pred,
        args.instruments,
        args.train_start,
        args.test_end,
        args.min_price,
        args.min_adv,
        args.max_volatility,
        args.liquidity_lookback,
        args.volatility_lookback,
    )
    pred = shift_signal_dates(pred, args.signal_shift_days)
    print(f"{args.model_type}_ts predict done, elapsed={time.perf_counter() - t_pred:.2f}s", flush=True)

    strategy = TopkDropoutStrategy(signal=pred, topk=args.topk, n_drop=args.n_drop)
    trade_executor = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
    t_bt = time.perf_counter()
    portfolio_metric_dict, _ = backtest(
        start_time=args.test_start,
        end_time=args.test_end,
        strategy=strategy,
        executor=trade_executor,
        benchmark=build_benchmark(args.benchmark.strip() or None, args.test_start, args.test_end),
        account=args.account,
        exchange_kwargs={
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": args.deal_price,
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    )
    report_df = portfolio_metric_dict["1day"][0]
    row = summarize_report(f"{args.model_type}_ts", report_df)
    row["fit_elapsed"] = fit_elapsed
    row["backtest_elapsed"] = time.perf_counter() - t_bt

    print(
        f"{args.model_type}_ts backtest done, "
        f"total_return={row['total_return']:.6f}, "
        f"annualized_return={row['annualized_return']:.6f}, "
        f"max_drawdown={row['max_drawdown']:.6f}",
        flush=True,
    )
    print("\nsummary:", flush=True)
    print(
        "total_return,annualized_return,information_ratio,max_drawdown,mean_turnover,total_cost,final_account,fit_elapsed,backtest_elapsed",
        flush=True,
    )
    print(
        "{total_return:.6f},{annualized_return:.6f},{information_ratio:.6f},{max_drawdown:.6f},"
        "{mean_turnover:.6f},{total_cost:.2f},{final_account:.2f},{fit_elapsed:.2f},{backtest_elapsed:.2f}".format(
            **row
        ),
        flush=True,
    )
    print(f"total elapsed={time.perf_counter() - wall_start:.2f}s", flush=True)


if __name__ == "__main__":
    main()

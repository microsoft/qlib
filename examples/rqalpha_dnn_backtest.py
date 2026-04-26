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

from rqalpha_lgbm_param_sweep import add_qlib_runtime_args, build_dataset_config, build_qlib_init_kwargs
from rqalpha_lgbm_preset_backtest import (
    apply_candidate_filters,
    apply_signal_filters,
    build_benchmark,
    shift_signal_dates,
    summarize_report,
)


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def ensure_torch_installed():
    if importlib.util.find_spec("torch") is None:
        raise SystemExit(
            "PyTorch is not installed in the current Python environment. "
            "Run `python -m pip install torch --index-url https://download.pytorch.org/whl/cu121` first."
        )


def clean_and_fill_features(train_x: np.ndarray, *other_arrays: np.ndarray, clip_value: float = 10.0):
    finite_train = np.where(np.isfinite(train_x), train_x, np.nan)
    feature_mean = np.nanmean(finite_train, axis=0)
    feature_mean = np.where(np.isfinite(feature_mean), feature_mean, 0.0).astype(np.float32)

    def _fill(array: np.ndarray):
        result = array.astype(np.float32, copy=True)
        bad_mask = ~np.isfinite(result)
        if bad_mask.any():
            row_idx, col_idx = np.where(bad_mask)
            result[row_idx, col_idx] = feature_mean[col_idx]
        if clip_value > 0:
            np.clip(result, -clip_value, clip_value, out=result)
        return result

    return (_fill(train_x), *[_fill(array) for array in other_arrays], feature_mean)


def main():
    parser = argparse.ArgumentParser(description="Run a GPU-heavier DNN stock-selection baseline on RQAlpha-converted Qlib data.")
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--train-start", default="2010-01-01")
    parser.add_argument("--train-end", default="2022-12-31")
    parser.add_argument("--valid-start", default="2023-01-01")
    parser.add_argument("--valid-end", default="2023-12-31")
    parser.add_argument("--test-start", default="2024-01-01")
    parser.add_argument("--test-end", default="2024-12-31")
    add_qlib_runtime_args(parser, default_device="gpu")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device id used by the DNN baseline.")
    parser.add_argument("--instruments", default="all")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--n-drop", type=int, default=3)
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
    parser.add_argument("--layers", default="1024,512,256", help="Comma-separated hidden sizes for the DNN.")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--early-stop-rounds", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--feature-clip", type=float, default=10.0)
    parser.add_argument("--grad-clip", type=float, default=3.0)
    parser.add_argument("--model-save-path", default="artifacts/rqalpha_dnn_best.pt")
    args = parser.parse_args()

    ensure_torch_installed()
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    if args.device != "gpu":
        raise SystemExit("This DNN baseline is designed for GPU usage. Please keep `--device gpu`.")

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
        build_dataset_config(
            args.instruments,
            args.train_start,
            args.train_end,
            args.valid_start,
            args.valid_end,
            args.test_start,
            args.test_end,
        )
    )
    print(f"dataset init done, elapsed={time.perf_counter() - t0:.2f}s", flush=True)

    t1 = time.perf_counter()
    train_df = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    valid_df = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    test_df = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
    print(f"prepare splits done, elapsed={time.perf_counter() - t1:.2f}s", flush=True)
    print(f"train shape={train_df.shape}", flush=True)
    print(f"valid shape={valid_df.shape}", flush=True)
    print(f"test shape={test_df.shape}", flush=True)

    train_x = train_df["feature"].values
    valid_x = valid_df["feature"].values
    test_x = test_df.values
    train_y = np.squeeze(train_df["label"].values).astype(np.float32)
    valid_y = np.squeeze(valid_df["label"].values).astype(np.float32)

    train_x, valid_x, test_x, _ = clean_and_fill_features(
        train_x,
        valid_x,
        test_x,
        clip_value=args.feature_clip,
    )
    train_y = np.nan_to_num(train_y, nan=0.0)
    valid_y = np.nan_to_num(valid_y, nan=0.0)

    feature_dim = train_x.shape[1]
    layers = parse_int_list(args.layers)
    print(f"feature_dim={feature_dim}", flush=True)
    print(f"layers={layers}", flush=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    modules = [nn.Dropout(0.05)]
    input_dim = feature_dim
    for hidden_dim in layers:
        modules.extend(
            [
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Dropout(0.05),
            ]
        )
        input_dim = hidden_dim
    modules.append(nn.Linear(input_dim, 1))
    model = nn.Sequential(*modules).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    valid_x_t = torch.from_numpy(valid_x).to(device)
    valid_y_t = torch.from_numpy(valid_y).to(device)
    model_save_path = Path(args.model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(args.gpu)

    t_fit = time.perf_counter()
    best_valid_loss = float("inf")
    best_step = 0
    stop_rounds = 0
    for step in range(1, args.max_steps + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(-1)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running_loss += float(loss.item())
            batch_count += 1

        if step % args.eval_steps == 0 or step == args.max_steps:
            model.eval()
            with torch.no_grad():
                valid_pred = model(valid_x_t)
                valid_loss = float(loss_fn(valid_pred, valid_y_t.unsqueeze(-1)).item())
            train_loss = running_loss / max(batch_count, 1)
            print(
                f"dnn step={step}, train_loss={train_loss:.6f}, valid_loss={valid_loss:.6f}",
                flush=True,
            )
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_step = step
                stop_rounds = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                stop_rounds += args.eval_steps
                if stop_rounds >= args.early_stop_rounds:
                    print(f"dnn early stop at step={step}", flush=True)
                    break

    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    fit_elapsed = time.perf_counter() - t_fit
    print(
        f"dnn fit done, elapsed={fit_elapsed:.2f}s, best_step={best_step}, best_valid_loss={best_valid_loss:.6f}",
        flush=True,
    )

    if torch.cuda.is_available():
        max_mem_gb = torch.cuda.max_memory_allocated(args.gpu) / (1024**3)
        print(f"torch_cuda_max_memory_allocated_gb={max_mem_gb:.2f}", flush=True)

    t_pred = time.perf_counter()
    model.eval()
    with torch.no_grad():
        pred_values = model(torch.from_numpy(test_x).to(device)).squeeze(-1).detach().cpu().numpy()
    pred = pd.Series(pred_values, index=test_df.index)
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
    print(f"dnn predict done, elapsed={time.perf_counter() - t_pred:.2f}s", flush=True)

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
    row = summarize_report("dnn", report_df)
    row["fit_elapsed"] = fit_elapsed
    row["backtest_elapsed"] = time.perf_counter() - t_bt

    print(
        "dnn backtest done, "
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

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
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
from qlib.constant import REG_CN
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.data import D
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config

from rqalpha_lgbm_param_sweep import (
    PRESETS,
    add_qlib_runtime_args,
    add_lgb_model_io_args,
    build_dataset_config,
    build_qlib_init_kwargs,
    load_lgb_model_artifact,
    parse_presets,
    save_lgb_model_artifact,
    split_feature_label,
)


def build_benchmark(benchmark: str | None, start_time: str, end_time: str):
    if benchmark:
        return benchmark
    calendar = pd.DatetimeIndex(D.calendar(start_time=start_time, end_time=end_time, freq="day"))
    return pd.Series(0.0, index=calendar)


def summarize_report(name: str, report_df: pd.DataFrame):
    total_return = report_df["account"].iloc[-1] / report_df["account"].iloc[0] - 1
    risk = risk_analysis(report_df["return"] - report_df["cost"], freq="day", mode="sum")
    return {
        "preset": name,
        "final_account": report_df["account"].iloc[-1],
        "total_return": total_return,
        "annualized_return": risk.loc["annualized_return", "risk"],
        "information_ratio": risk.loc["information_ratio", "risk"],
        "max_drawdown": risk.loc["max_drawdown", "risk"],
        "mean_turnover": report_df["turnover"].mean(),
        "total_cost": report_df["total_cost"].iloc[-1],
    }


def apply_signal_filters(
    signal: pd.Series,
    instruments: str,
    start_time: str,
    end_time: str,
    min_history_days: int,
) -> pd.Series:
    if min_history_days <= 0:
        return signal

    # This is a lightweight guardrail against newly listed names dominating
    # research backtests with short, noisy histories.
    fields = ["$close"]
    close = D.features(D.instruments(instruments), fields, start_time=start_time, end_time=end_time, freq="day")
    if isinstance(close.index, pd.MultiIndex) and close.index.names[:2] != ["datetime", "instrument"]:
        close = close.reorder_levels(["datetime", "instrument"]).sort_index()
    close = close.rename(columns={"$close": "close"})
    close["history_days"] = close["close"].notna().groupby(level="instrument").cumsum()
    history_days = close["history_days"].reindex(signal.index)
    filtered = signal.mask(history_days < min_history_days)
    dropped = int(signal.notna().sum() - filtered.notna().sum())
    print(f"signal filter: min_history_days={min_history_days}, dropped={dropped}", flush=True)
    return filtered


def shift_signal_dates(signal: pd.Series, shift_days: int) -> pd.Series:
    if shift_days <= 0:
        return signal
    if not isinstance(signal.index, pd.MultiIndex) or signal.index.names[:2] != ["datetime", "instrument"]:
        raise ValueError("signal index must be a MultiIndex with datetime and instrument levels")

    dates = pd.DatetimeIndex(signal.index.get_level_values("datetime"))
    calendar = pd.DatetimeIndex(D.calendar(start_time=dates.min(), end_time=dates.max(), freq="day"))
    shift_map = {
        calendar[i]: calendar[i + shift_days]
        for i in range(0, max(len(calendar) - shift_days, 0))
    }
    shifted_dates = dates.map(shift_map)
    keep = ~pd.isna(shifted_dates)
    shifted = signal.loc[keep].copy()
    shifted.index = pd.MultiIndex.from_arrays(
        [
            pd.DatetimeIndex(shifted_dates[keep]),
            shifted.index.get_level_values("instrument"),
        ],
        names=signal.index.names,
    )
    print(f"signal shift: shift_days={shift_days}, dropped={int((~keep).sum())}", flush=True)
    return shifted


def build_candidate_mask(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    min_price: float,
    min_adv: float,
    max_volatility: float,
    liquidity_lookback: int,
    volatility_lookback: int,
) -> pd.DataFrame:
    mask = close.notna() & volume.notna()

    if min_price > 0:
        mask &= close >= min_price

    if min_adv > 0:
        adv = (close * volume).rolling(liquidity_lookback, min_periods=liquidity_lookback).mean()
        mask &= adv >= min_adv

    if max_volatility > 0:
        volatility = close.pct_change().rolling(
            volatility_lookback,
            min_periods=volatility_lookback,
        ).std()
        mask &= volatility <= max_volatility

    return mask


def apply_candidate_filters(
    signal: pd.Series,
    instruments: str,
    start_time: str,
    end_time: str,
    min_price: float,
    min_adv: float,
    max_volatility: float,
    liquidity_lookback: int,
    volatility_lookback: int,
) -> pd.Series:
    if min_price <= 0 and min_adv <= 0 and max_volatility <= 0:
        return signal

    fields = ["$close", "$volume"]
    market = D.features(D.instruments(instruments), fields, start_time=start_time, end_time=end_time, freq="day")
    if isinstance(market.index, pd.MultiIndex) and market.index.names[:2] != ["datetime", "instrument"]:
        market = market.reorder_levels(["datetime", "instrument"]).sort_index()

    close = market["$close"].unstack("instrument").sort_index()
    volume = market["$volume"].unstack("instrument").sort_index()
    mask = build_candidate_mask(
        close,
        volume,
        min_price=min_price,
        min_adv=min_adv,
        max_volatility=max_volatility,
        liquidity_lookback=liquidity_lookback,
        volatility_lookback=volatility_lookback,
    )
    mask_series = mask.stack(dropna=False).reindex(signal.index)
    filtered = signal.where(mask_series.fillna(False))
    dropped = int(signal.notna().sum() - filtered.notna().sum())
    print(
        "candidate filter: "
        f"min_price={min_price}, min_adv={min_adv}, max_volatility={max_volatility}, dropped={dropped}",
        flush=True,
    )
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Backtest LightGBM presets on RQAlpha-converted Qlib data.")
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--train-start", default="2010-01-01")
    parser.add_argument("--train-end", default="2018-12-31")
    parser.add_argument("--valid-start", default="2019-01-01")
    parser.add_argument("--valid-end", default="2019-12-31")
    parser.add_argument("--test-start", default="2020-01-01")
    parser.add_argument("--test-end", default="2020-12-31")
    add_qlib_runtime_args(parser, default_device="gpu")
    parser.add_argument("--instruments", default="all")
    parser.add_argument("--presets", default="baseline,wider")
    parser.add_argument("--num-boost-round", type=int, default=1000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    add_lgb_model_io_args(parser)
    parser.add_argument("--topk", type=int, default=80)
    parser.add_argument("--n-drop", type=int, default=10)
    parser.add_argument("--benchmark", default="", help="Optional benchmark instrument such as SH000300.")
    parser.add_argument("--deal-price", default="close", help="Backtest deal price field, such as close, open, or vwap.")
    parser.add_argument(
        "--signal-shift-days",
        type=int,
        default=1,
        help="Move prediction timestamps forward by N trading days before backtesting.",
    )
    parser.add_argument(
        "--min-history-days",
        type=int,
        default=252,
        help="Mask predictions for instruments with fewer valid close-price days since train-start.",
    )
    parser.add_argument("--min-price", type=float, default=0.0, help="Mask instruments below this close price.")
    parser.add_argument(
        "--min-adv",
        type=float,
        default=0.0,
        help="Mask instruments whose rolling average daily turnover is below this threshold.",
    )
    parser.add_argument(
        "--max-volatility",
        type=float,
        default=0.0,
        help="Mask instruments whose rolling daily close volatility exceeds this threshold.",
    )
    parser.add_argument("--liquidity-lookback", type=int, default=20)
    parser.add_argument("--volatility-lookback", type=int, default=20)
    args = parser.parse_args()

    provider_uri = str(Path(args.provider_uri).expanduser().resolve())
    preset_names = parse_presets(args.presets)

    wall_start = time.perf_counter()
    print(f"provider_uri={provider_uri}", flush=True)
    print(f"presets={','.join(preset_names)}", flush=True)
    print(f"device={args.device}", flush=True)
    print(f"dataset_cache={args.dataset_cache}", flush=True)
    print(f"load_models={args.load_models}", flush=True)
    print(f"save_models={args.save_models}", flush=True)
    print(f"model_dir={Path(args.model_dir).expanduser().resolve()}", flush=True)
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

    train_x, train_y = split_feature_label(train_df)
    valid_x, valid_y = split_feature_label(valid_df)
    test_x = test_df.values
    train_set = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
    valid_set = lgb.Dataset(valid_x, label=valid_y, reference=train_set, free_raw_data=False)
    benchmark = build_benchmark(args.benchmark.strip() or None, args.test_start, args.test_end)

    rows = []
    for name in preset_names:
        fit_elapsed = 0.0
        pred_iteration = None
        if args.load_models:
            print(f"\n[{name}] load model start", flush=True)
            model, metadata, model_path, _meta_path = load_lgb_model_artifact(args, name)
            valid_l2 = float(metadata.get("valid_l2", float("nan")))
            pred_iteration = int(metadata["best_iteration"]) if metadata.get("best_iteration") is not None else None
            print(
                f"[{name}] load model done, best_iteration={pred_iteration}, "
                f"valid_l2={valid_l2:.6f}, model_path={model_path}",
                flush=True,
            )
        else:
            params = {
                "objective": "mse",
                "metric": "l2",
                "verbosity": -1,
                "num_threads": args.num_threads,
                "device": args.device,
                **PRESETS[name],
            }
            evals_result = {}
            print(f"\n[{name}] train start", flush=True)
            t_fit = time.perf_counter()
            model = lgb.train(
                params,
                train_set,
                num_boost_round=args.num_boost_round,
                valid_sets=[train_set, valid_set],
                valid_names=["train", "valid"],
                callbacks=[
                    lgb.early_stopping(args.early_stopping_rounds),
                    lgb.log_evaluation(period=20),
                    lgb.record_evaluation(evals_result),
                ],
            )
            fit_elapsed = time.perf_counter() - t_fit
            valid_l2 = min(evals_result["valid"]["l2"])
            pred_iteration = model.best_iteration
            print(f"[{name}] train done, best_iteration={model.best_iteration}, valid_l2={valid_l2:.6f}", flush=True)
            if args.save_models:
                model_path, _meta_path = save_lgb_model_artifact(
                    model,
                    args,
                    name,
                    extra_meta={"valid_l2": valid_l2},
                )
                print(f"[{name}] model saved to {model_path}", flush=True)

        t_pred = time.perf_counter()
        pred = pd.Series(model.predict(test_x, num_iteration=pred_iteration), index=test_df.index)
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
        print(f"[{name}] predict done, elapsed={time.perf_counter() - t_pred:.2f}s", flush=True)

        t_bt = time.perf_counter()
        strategy = TopkDropoutStrategy(signal=pred, topk=args.topk, n_drop=args.n_drop)
        trade_executor = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
        portfolio_metric_dict, _ = backtest(
            start_time=args.test_start,
            end_time=args.test_end,
            strategy=strategy,
            executor=trade_executor,
            benchmark=benchmark,
            account=10000000,
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
        row = summarize_report(name, report_df)
        row["best_iteration"] = pred_iteration if pred_iteration is not None else -1
        row["valid_l2"] = valid_l2
        row["fit_elapsed"] = fit_elapsed
        row["backtest_elapsed"] = time.perf_counter() - t_bt
        rows.append(row)
        print(
            f"[{name}] backtest done, total_return={row['total_return']:.6f}, "
            f"annualized_return={row['annualized_return']:.6f}, "
            f"max_drawdown={row['max_drawdown']:.6f}",
            flush=True,
        )

    print("\nsummary:", flush=True)
    print(
        "preset,best_iteration,valid_l2,total_return,annualized_return,information_ratio,"
        "max_drawdown,mean_turnover,total_cost,final_account,fit_elapsed,backtest_elapsed",
        flush=True,
    )
    for row in sorted(rows, key=lambda item: item["total_return"], reverse=True):
        print(
            "{preset},{best_iteration},{valid_l2:.6f},{total_return:.6f},{annualized_return:.6f},"
            "{information_ratio:.6f},{max_drawdown:.6f},{mean_turnover:.6f},{total_cost:.2f},"
            "{final_account:.2f},{fit_elapsed:.2f},{backtest_elapsed:.2f}".format(**row),
            flush=True,
        )
    print(f"total elapsed={time.perf_counter() - wall_start:.2f}s", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
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
from rqalpha_lgbm_preset_backtest import (
    apply_candidate_filters,
    apply_signal_filters,
    build_benchmark,
    shift_signal_dates,
    summarize_report,
)


def calculate_position_contribution(
    positions: dict,
    start_time: str,
    end_time: str,
    init_account: float,
) -> pd.DataFrame:
    dates = sorted(pd.Timestamp(date) for date in positions)
    stocks = sorted({stock for position in positions.values() for stock in position.get_stock_list()})
    if not dates or not stocks:
        return pd.DataFrame(columns=["contribution", "hold_days", "contribution_pct_init"])

    close = D.features(stocks, ["$close"], start_time=start_time, end_time=end_time, freq="day")
    if isinstance(close.index, pd.MultiIndex) and close.index.names[:2] != ["datetime", "instrument"]:
        close = close.reorder_levels(["datetime", "instrument"]).sort_index()
    close = close["$close"].unstack("instrument").sort_index()

    # End-of-day position on T earns the close-to-close return from T to T+1.
    next_day_return = close.pct_change().shift(-1)
    contribution: dict[str, float] = {}
    hold_days: dict[str, int] = {}

    for date in dates:
        if date not in next_day_return.index:
            continue
        row_return = next_day_return.loc[date]
        account_value = positions[date].calculate_value()
        for stock, weight in positions[date].get_stock_weight_dict().items():
            stock_return = row_return.get(stock)
            if pd.isna(stock_return):
                continue
            contribution[stock] = contribution.get(stock, 0.0) + account_value * weight * float(stock_return)
            hold_days[stock] = hold_days.get(stock, 0) + 1

    result = pd.DataFrame(
        {
            "contribution": pd.Series(contribution),
            "hold_days": pd.Series(hold_days),
        }
    ).dropna()
    result["contribution_pct_init"] = result["contribution"] / init_account
    return result


def main():
    parser = argparse.ArgumentParser(description="Rank stock-level contribution for an RQAlpha/Qlib backtest.")
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--train-start", default="2010-01-01")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--valid-start", default="2024-01-01")
    parser.add_argument("--valid-end", default="2024-12-31")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--test-end", default="2025-12-31")
    add_qlib_runtime_args(parser, default_device="gpu")
    parser.add_argument("--instruments", default="all")
    parser.add_argument("--preset", default="baseline")
    parser.add_argument("--num-boost-round", type=int, default=120)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    add_lgb_model_io_args(parser)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--n-drop", type=int, default=5)
    parser.add_argument("--min-history-days", type=int, default=252)
    parser.add_argument("--signal-shift-days", type=int, default=1)
    parser.add_argument("--min-price", type=float, default=0.0)
    parser.add_argument("--min-adv", type=float, default=0.0)
    parser.add_argument("--max-volatility", type=float, default=0.0)
    parser.add_argument("--liquidity-lookback", type=int, default=20)
    parser.add_argument("--volatility-lookback", type=int, default=20)
    parser.add_argument("--deal-price", default="close")
    parser.add_argument("--account", type=float, default=10000000)
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    preset = parse_presets(args.preset)[0]
    wall_start = time.perf_counter()

    print("[1/7] qlib init", flush=True)
    provider_uri = str(Path(args.provider_uri).expanduser().resolve())
    print(f"dataset_cache={args.dataset_cache}", flush=True)
    print(f"load_models={args.load_models}", flush=True)
    print(f"save_models={args.save_models}", flush=True)
    print(f"model_dir={Path(args.model_dir).expanduser().resolve()}", flush=True)
    if args.dataset_cache == "simple":
        print(f"local_cache_path={Path(args.local_cache_path).expanduser().resolve()}", flush=True)
    qlib.init(**build_qlib_init_kwargs(args, provider_uri))

    print("[2/7] building dataset", flush=True)
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
    print(f"[2/7] dataset ready, elapsed={time.perf_counter() - t0:.2f}s", flush=True)

    print("[3/7] preparing train/valid/test arrays", flush=True)
    t1 = time.perf_counter()
    train_df = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    valid_df = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    test_df = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
    print(
        f"[3/7] arrays ready, train={train_df.shape}, valid={valid_df.shape}, "
        f"test={test_df.shape}, elapsed={time.perf_counter() - t1:.2f}s",
        flush=True,
    )

    train_x, train_y = split_feature_label(train_df)
    valid_x, valid_y = split_feature_label(valid_df)
    train_set = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
    valid_set = lgb.Dataset(valid_x, label=valid_y, reference=train_set, free_raw_data=False)
    pred_iteration = None
    if args.load_models:
        print(f"[4/7] loading LightGBM preset={preset}", flush=True)
        model, metadata, model_path, _meta_path = load_lgb_model_artifact(args, preset)
        valid_l2 = float(metadata.get("valid_l2", float("nan")))
        pred_iteration = int(metadata["best_iteration"]) if metadata.get("best_iteration") is not None else None
        print(
            f"[4/7] model loaded, best_iteration={pred_iteration}, valid_l2={valid_l2:.6f}, model_path={model_path}",
            flush=True,
        )
    else:
        print(f"[4/7] training LightGBM preset={preset}", flush=True)
        params = {
            "objective": "mse",
            "metric": "l2",
            "verbosity": -1,
            "num_threads": args.num_threads,
            "device": args.device,
            **PRESETS[preset],
        }
        evals_result: dict = {}
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
        valid_l2 = min(evals_result["valid"]["l2"])
        pred_iteration = model.best_iteration
        print(f"[4/7] model ready, best_iteration={pred_iteration}, valid_l2={valid_l2:.6f}", flush=True)
        if args.save_models:
            model_path, _meta_path = save_lgb_model_artifact(
                model,
                args,
                preset,
                extra_meta={"valid_l2": valid_l2},
            )
            print(f"[4/7] model saved to {model_path}", flush=True)

    print("[5/7] predicting and applying signal filters", flush=True)
    pred = pd.Series(model.predict(test_df.values, num_iteration=pred_iteration), index=test_df.index)
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

    print("[6/7] running backtest", flush=True)
    strategy = TopkDropoutStrategy(signal=pred, topk=args.topk, n_drop=args.n_drop)
    executor = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
    portfolio_metric_dict, _ = backtest(
        start_time=args.test_start,
        end_time=args.test_end,
        strategy=strategy,
        executor=executor,
        benchmark=build_benchmark(None, args.test_start, args.test_end),
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
    report_df, positions = portfolio_metric_dict["1day"]
    summary = summarize_report(f"{preset}_shift{args.signal_shift_days}", report_df)
    print(
        "[6/7] backtest ready, total_return={total_return:.6f}, "
        "annualized_return={annualized_return:.6f}, max_drawdown={max_drawdown:.6f}".format(**summary),
        flush=True,
    )

    print("[7/7] calculating stock contribution", flush=True)
    contribution = calculate_position_contribution(positions, args.test_start, args.test_end, args.account)
    winners = contribution.sort_values("contribution", ascending=False).head(args.top_n)
    losers = contribution.sort_values("contribution", ascending=True).head(args.top_n)

    print("\nwinners:", flush=True)
    print(winners.to_string(float_format=lambda value: f"{value:.6f}"), flush=True)
    print("\nlosers:", flush=True)
    print(losers.to_string(float_format=lambda value: f"{value:.6f}"), flush=True)
    print(f"\ntotal elapsed={time.perf_counter() - wall_start:.2f}s", flush=True)


if __name__ == "__main__":
    main()

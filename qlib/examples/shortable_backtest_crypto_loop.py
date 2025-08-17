import os
import pandas as pd
import qlib
from qlib.data import D
from qlib.constant import REG_CRYPTO

from qlib.backtest.shortable_backtest import ShortableExecutor, ShortableAccount
from qlib.backtest.shortable_exchange import ShortableExchange
from qlib.backtest.decision import OrderDir
from qlib.contrib.strategy.signal_strategy import LongShortTopKStrategy


def main():
    provider = os.path.expanduser("~/.qlib/qlib_data/crypto_data_perp")
    qlib.init(provider_uri=provider, region=REG_CRYPTO, kernels=1)

    start = pd.Timestamp("2021-07-11")
    end = pd.Timestamp("2021-08-10")

    # Universe
    inst_conf = D.instruments("all")
    codes = D.list_instruments(inst_conf, start_time=start, end_time=end, freq="day", as_list=True)[:20]
    if not codes:
        print("No instruments.")
        return

    # Exchange
    ex = ShortableExchange(
        freq="day",
        start_time=start,
        end_time=end,
        codes=codes,
        deal_price="$close",
        open_cost=0.0005,
        close_cost=0.0015,
        min_cost=0.0,
        impact_cost=0.0,
        limit_threshold=None,
    )

    # Account and executor
    account = ShortableAccount(benchmark_config={"benchmark": None})
    exe = ShortableExecutor(
        time_per_step="day",
        generate_portfolio_metrics=True,
        trade_exchange=ex,
        region="crypto",
        verbose=False,
        account=account,
    )
    exe.reset(start_time=start, end_time=end)

    # Precompute momentum signal for the whole period (shift=1 used by strategy)
    feat = D.features(codes, ["$close"], start, end, freq="day", disk_cache=True)
    if feat is None or feat.empty:
        print("No features to build signal.")
        return
    feat = feat.sort_index()
    grp = feat.groupby("instrument")["$close"]
    prev_close = grp.shift(1)
    mom = (feat["$close"] / prev_close - 1.0).rename("score")
    # Use MultiIndex Series (instrument, datetime)
    signal_series = mom.dropna()

    # Strategy (TopK-aligned, long-short)
    strat = LongShortTopKStrategy(
        topk_long=3,
        topk_short=3,
        n_drop_long=1,
        n_drop_short=1,
        only_tradable=False,
        forbid_all_trade_at_limit=True,
        signal=signal_series,
        trade_exchange=ex,
    )
    # Bind strategy infra to executor
    strat.reset(level_infra=exe.get_level_infra(), common_infra=exe.common_infra)

    # Drive by executor calendar
    while not exe.finished():
        td = strat.generate_trade_decision()
        exe.execute(td)

    # Output metrics
    df, meta = exe.trade_account.get_portfolio_metrics()
    print("Portfolio metrics meta:", meta)
    print("Portfolio df tail:\n", df.tail() if hasattr(df, "tail") else df)


if __name__ == "__main__":
    main()



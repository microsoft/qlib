import os
import pandas as pd
import qlib
from qlib.data import D
from qlib.constant import REG_CRYPTO

from qlib.backtest.shortable_backtest import ShortableExecutor, LongShortStrategy, ShortableAccount
from qlib.backtest.shortable_exchange import ShortableExchange
from qlib.backtest.decision import OrderDir


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

    # Strategy
    strat = LongShortStrategy(gross_leverage=1.0, net_exposure=0.0, top_k=3, exchange=ex,
                              lot_size=None, min_trade_threshold=None)

    # Drive by executor calendar to ensure alignment
    while not exe.finished():
        d, _ = exe.trade_calendar.get_step_time()
        # Build simple momentum signal (last/prev - 1); fallback to last close demean
        feat = D.features(codes, ["$close"], d - pd.Timedelta(days=10), d, freq="day", disk_cache=True)
        if feat is None or feat.empty:
            td = strat.generate_trade_decision(pd.Series(dtype=float), exe.position, d)
            exe.execute(td)
            continue
        g = feat.groupby("instrument")["$close"]
        last = g.last()
        # robust prev: each group iloc[-2]
        try:
            prev = g.apply(lambda s: s.iloc[-2])
            sig = (last / prev - 1.0).dropna()
        except Exception:
            sig = pd.Series(dtype=float)
        if sig.empty:
            last = last.dropna()
            sig = (last - last.mean()) if not last.empty else pd.Series(dtype=float)

        td = strat.generate_trade_decision(sig, exe.position, d)
        exe.execute(td)

    # Output metrics
    df, meta = exe.trade_account.get_portfolio_metrics()
    print("Portfolio metrics meta:", meta)
    print("Portfolio df tail:\n", df.tail() if hasattr(df, "tail") else df)


if __name__ == "__main__":
    main()



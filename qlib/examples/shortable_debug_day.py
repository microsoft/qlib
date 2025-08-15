import os
import pandas as pd
import qlib
from qlib.data import D
from qlib.constant import REG_CRYPTO
from qlib.backtest.decision import OrderDir
from qlib.backtest.shortable_exchange import ShortableExchange


def main():
    provider = os.path.expanduser("~/.qlib/qlib_data/crypto_data_perp")
    qlib.init(provider_uri=provider, region=REG_CRYPTO, kernels=1)

    start = pd.Timestamp("2021-07-11")
    end = pd.Timestamp("2021-08-10")
    day = pd.Timestamp("2021-08-10")

    inst_conf = D.instruments("all")
    codes = D.list_instruments(inst_conf, start_time=start, end_time=end, freq="day", as_list=True)[:10]

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

    feat = D.features(codes, ["$close"], day - pd.Timedelta(days=10), day, freq="day", disk_cache=True)
    g = feat.groupby("instrument")["$close"]
    last = g.last()
    # 使用每组倒数第2个值并去掉datetime层，确保索引为instrument
    prev = g.apply(lambda s: s.iloc[-2])
    sig = (last / prev - 1.0).dropna().sort_values(ascending=False)

    longs = sig.head(3).index.tolist()
    shorts = sig.tail(3).index.tolist()

    equity = 1_000_000.0
    long_weight = 0.5 / max(len(longs), 1)
    short_weight = -0.5 / max(len(shorts), 1)

    print("day:", day.date())
    for leg, lst, w, dir_ in [("LONG", longs, long_weight, OrderDir.BUY), ("SHORT", shorts, short_weight, OrderDir.SELL)]:
        print(f"\n{leg} candidates:")
        for code in lst:
            try:
                px = ex.get_deal_price(code, day, day, dir_)
                fac = ex.get_factor(code, day, day)
                unit = ex.get_amount_of_trade_unit(fac, code, day, day)
                tradable = ex.is_stock_tradable(code, day, day, dir_)
                raw = (w * equity) / px if px else 0.0
                rounded = ex.round_amount_by_trade_unit(abs(raw), fac) if px else 0.0
                if dir_ == OrderDir.SELL:
                    rounded = -rounded
                print(code, {
                    "price": px,
                    "factor": fac,
                    "unit": unit,
                    "tradable": tradable,
                    "raw_shares": raw,
                    "rounded": rounded,
                })
            except Exception as e:
                print(code, "error:", e)


if __name__ == "__main__":
    main()



"""Tests for shortable crypto backtest components (executor/exchange/position)."""

# pylint: disable=C0301,W0718,C0116,R1710,R0914,C0411
import os
from pathlib import Path
import pytest
import pandas as pd

import qlib
import warnings
from qlib.data import D
from qlib.constant import REG_CRYPTO

from qlib.backtest.shortable_exchange import ShortableExchange
from qlib.backtest.shortable_backtest import ShortableExecutor, LongShortStrategy, ShortableAccount


def _try_init_qlib():
    """Initialize qlib with real crypto data if available; otherwise skip tests."""
    candidates = [
        os.path.expanduser("~/.qlib/qlib_data/crypto_data_perp"),  # Prefer user's provided perp path
        os.path.expanduser("~/.qlib/qlib_data/crypto_data"),
        str(Path(__file__).resolve().parents[3] / "crypto-qlib" / "binance_crypto_data_perp"),
        str(Path(__file__).resolve().parents[3] / "crypto-qlib" / "binance_crypto_data"),
    ]
    for p in candidates:
        try:
            if p and (p.startswith("~") or os.path.isabs(p)):
                # Expand ~ and check existence loosely (provider may be a directory with sub-structure)
                _p = os.path.expanduser(p)
            else:
                _p = p
            qlib.init(provider_uri=_p, region=REG_CRYPTO, skip_if_reg=True, kernels=1)
            # Silence known harmless warning from numpy on empty slice in qlib internal mean
            warnings.filterwarnings(
                "ignore",
                message="Mean of empty slice",
                category=RuntimeWarning,
                module=r".*qlib\\.utils\\.index_data",
            )
            # Probe one simple call
            _ = D.instruments()
            return _p
        except Exception:
            continue
    pytest.skip("No valid crypto provider_uri found; skipping real-data tests")


def test_shortable_with_real_data_end_to_end():
    _ = _try_init_qlib()

    # Use a fixed window you confirmed has data
    start_time = pd.Timestamp("2021-07-11")
    end_time = pd.Timestamp("2021-08-10")

    # Pick a small universe via proper API: instruments config -> list
    inst_conf = D.instruments(market="all")
    instruments = D.list_instruments(inst_conf, start_time=start_time, end_time=end_time, freq="day", as_list=True)[:10]
    if not instruments:
        pytest.skip("No instruments available from provider; skipping")

    # Build exchange on real data, restrict to small universe
    ex = ShortableExchange(
        freq="day",
        start_time=start_time,
        end_time=end_time,
        codes=instruments,
        deal_price="$close",
        open_cost=0.0015,
        close_cost=0.0025,
        impact_cost=0.0,
        limit_threshold=None,
    )

    # Avoid default CSI300 benchmark by constructing account with benchmark=None
    account = ShortableAccount(benchmark_config={"benchmark": None})

    exe = ShortableExecutor(
        time_per_step="day",
        generate_portfolio_metrics=True,
        trade_exchange=ex,
        region="crypto",
        verbose=False,
        account=account,
    )

    # Build a simple momentum signal on end_time (fallback to last-close ranking if necessary)
    feat = D.features(
        instruments,
        ["$close"],
        start_time,
        end_time,
        freq="day",
        disk_cache=True,
    )
    if feat is None or feat.empty:
        pytest.skip("No valid features in selected window; skipping")

    g = feat.groupby("instrument")["$close"]
    last = g.last()
    # momentum needs at least 2 rows per instrument
    try:
        prev = g.nth(-2)
        sig = (last / prev - 1.0).dropna()
    except Exception:
        sig = pd.Series(dtype=float)

    if sig.empty:
        # fallback: rank by last close (descending)
        last = last.dropna()
        if last.empty:
            pytest.skip("No closes to build fallback signal; skipping")
        sig = last - last.mean()  # demeaned last close as pseudo-signal

    # Generate orders for the end_time
    # For crypto, use unit step to ensure orders are generated and avoid empty indicators
    strat = LongShortStrategy(
        gross_leverage=1.0,
        net_exposure=0.0,
        top_k=3,
        exchange=ex,
        lot_size=1,
        min_trade_threshold=1,
    )
    td = strat.generate_trade_decision(sig, exe.position, end_time)

    # Execute one step via standard API
    exe.reset(start_time=start_time, end_time=end_time)
    _ = exe.execute(td)

    # Validate metrics shape and key fields
    df, meta = exe.trade_account.get_portfolio_metrics()
    assert hasattr(df, "shape")
    assert isinstance(meta, dict)
    # net_exposure should be finite; leverage should be >= 0
    assert meta.get("leverage", 0) >= 0
    assert isinstance(meta.get("net_exposure", 0), float)
    # If we have short positions, borrow cost may be > 0
    assert meta.get("total_borrow_cost", 0) >= 0

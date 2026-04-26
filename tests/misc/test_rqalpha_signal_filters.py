import pandas as pd

from examples.rqalpha_lgbm_preset_backtest import build_candidate_mask


def test_build_candidate_mask_combines_price_liquidity_and_volatility():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    close = pd.DataFrame(
        {
            "AAA": [10.0, 10.2, 10.3, 10.4],
            "BBB": [3.0, 3.1, 3.2, 3.3],
            "CCC": [10.0, 15.0, 8.0, 16.0],
        },
        index=dates,
    )
    volume = pd.DataFrame(
        {
            "AAA": [5_000_000, 5_000_000, 5_000_000, 5_000_000],
            "BBB": [5_000_000, 5_000_000, 5_000_000, 5_000_000],
            "CCC": [5_000_000, 5_000_000, 5_000_000, 5_000_000],
        },
        index=dates,
    )

    mask = build_candidate_mask(
        close,
        volume,
        min_price=5.0,
        min_adv=40_000_000.0,
        max_volatility=0.4,
        liquidity_lookback=2,
        volatility_lookback=2,
    )

    assert bool(mask.loc[dates[-1], "AAA"])
    assert not bool(mask.loc[dates[-1], "BBB"])
    assert not bool(mask.loc[dates[-1], "CCC"])

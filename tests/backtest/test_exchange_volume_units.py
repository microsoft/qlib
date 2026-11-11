"""Offline regression coverage for the price/volume/order unit contract (#2362)."""

import numpy as np
import pytest

from qlib.backtest.decision import Order
from qlib.backtest.exchange import Exchange


@pytest.fixture
def volume_market(workflow_context, request):
    """Represent one stock-day as shares, converted lots, and unconverted lots."""
    factor = request.param
    symbols = ["SH600001", "SH600002", "SH600003"]
    raw_share_volume = 100_000
    raw_lot_volume = raw_share_volume / 100
    volumes = [raw_share_volume, raw_lot_volume * 100, raw_lot_volume]
    for symbol, volume in zip(symbols, volumes):
        directory = workflow_context.provider_uri / "features" / symbol.lower()
        directory.mkdir(exist_ok=True)
        for field, value in {
            "close": 20 * factor,
            "factor": factor,
            "volume": volume / factor,
            "change": 0,
        }.items():
            np.array([0, value], dtype="<f4").tofile(directory / f"{field}.day.bin")
    return workflow_context.calendar[0], symbols, factor


def execute(date, symbols, factor, direction, amount, volume_threshold=None):
    exchange = Exchange(
        codes=symbols,
        start_time=date,
        end_time=date,
        deal_price="$close",
        limit_threshold=0.99,
        volume_threshold=volume_threshold,
        trade_unit=None,
        open_cost=0,
        close_cost=0,
        min_cost=0,
        impact_cost=0.1,
    )
    results = []
    for symbol in symbols:
        order = Order(symbol, amount / factor, direction, date, date)
        trade_val, trade_cost, trade_price = exchange.deal_order(order)
        results.append((order.deal_amount * factor, trade_val, trade_cost, trade_price))
    return results


@pytest.mark.parametrize("volume_market", [1.0, 2.0, 0.05], indirect=True)
@pytest.mark.parametrize("direction", [Order.BUY, Order.SELL])
def test_impact_cost_requires_volume_in_adjusted_order_units(volume_market, direction):
    date, symbols, factor = volume_market
    shares, converted_lots, unconverted_lots = execute(date, symbols, factor, direction, amount=2000)
    # A 2% participation rate gives 0.1 * 0.02**2 * 40,000 = 1.6 cost.
    assert shares[:3] == pytest.approx((2000, 40_000, 1.6))
    assert converted_lots == pytest.approx(shares)
    # Omitting the lot-to-share conversion understates volume by 100, so
    # quadratic impact cost is inflated by 100**2. No regional correction is inferred.
    assert unconverted_lots[:2] == pytest.approx(shares[:2])
    assert unconverted_lots[2] == pytest.approx(shares[2] * 10_000)


@pytest.mark.parametrize("volume_market", [1.0, 2.0, 0.05], indirect=True)
@pytest.mark.parametrize("direction", [Order.BUY, Order.SELL])
def test_capacity_limit_requires_volume_in_adjusted_order_units(volume_market, direction):
    date, symbols, factor = volume_market
    shares, converted_lots, unconverted_lots = execute(
        date, symbols, factor, direction, amount=50_000, volume_threshold=("current", "0.1 * $volume")
    )
    assert shares[:3] == pytest.approx((10_000, 200_000, 200))
    assert converted_lots == pytest.approx(shares)
    # With the same erroneous volume in both paths, the limit fills 100x fewer
    # shares while the apparent participation still equals the configured 10%.
    assert unconverted_lots[:3] == pytest.approx((100, 2000, 2))

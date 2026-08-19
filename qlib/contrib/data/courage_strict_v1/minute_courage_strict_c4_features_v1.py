"""Pure feature kernels for the Courage-strict C4 main-board route."""

from __future__ import annotations

import warnings
from typing import Final

import numpy as np
import pandas as pd


class CourageStrictC4FeatureError(RuntimeError):
    """Raised when a C4 feature input or output violates the frozen contract."""


DYNAMIC_FEATURES: Final[tuple[str, ...]] = (
    "stock_ret_1",
    "stock_ret_5",
    "stock_ret_20",
    "stock_vwap_bias_20",
    "stock_realized_vol_20",
    "stock_high_low_range_20",
    "stock_volume_ratio_20",
    "stock_amount_ratio_20",
    "sector_ret_5_median",
    "sector_ret_20_median",
    "sector_positive_breadth_1",
    "sector_amount_ratio_20_median",
)
STOCK_FEATURES: Final[tuple[str, ...]] = DYNAMIC_FEATURES[:8]
SECTOR_FEATURES: Final[tuple[str, ...]] = DYNAMIC_FEATURES[8:]
SLOW_FEATURES: Final[tuple[str, ...]] = (
    "daily_ret_5",
    "daily_ma20_bias",
    "daily_vol_20",
    "turnover_mean_60",
    "market_cap_log",
)
SECTOR_SOURCE_INDEX: Final[dict[str, int]] = {
    "sector_ret_5_median": 1,
    "sector_ret_20_median": 2,
    "sector_positive_breadth_1": 0,
    "sector_amount_ratio_20_median": 7,
}


def _last_mark_age(
    length: int, offsets: np.ndarray, *, initial_offset: int
) -> np.ndarray:
    marks = np.full(length, int(initial_offset), dtype=np.int64)
    valid = np.asarray(offsets, dtype=np.int64)
    valid = valid[(valid >= 0) & (valid < length)]
    marks[valid] = valid
    return np.arange(length, dtype=np.int64) - np.maximum.accumulate(marks)


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    prefix = np.concatenate(([0.0], np.cumsum(source)))
    result = np.full(len(source), np.nan, dtype=np.float64)
    result[window - 1 :] = prefix[window:] - prefix[:-window]
    return result


def _rolling_max_min(values: np.ndarray, window: int, mode: str) -> np.ndarray:
    source = pd.Series(np.asarray(values, dtype=np.float64))
    rolling = source.rolling(window=window, min_periods=window)
    return (rolling.max() if mode == "max" else rolling.min()).to_numpy(
        dtype=np.float64
    )


def compute_stock_features_v1(
    *,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    amount: np.ndarray,
    event_offsets: np.ndarray,
    quarantine_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``values, available, missing`` for the eight stock features.

    All arrays follow one dense official-minute axis.  A usable source bar must
    have finite positive OHLC, volume, and amount.  Accepted and quarantined
    action dates reset rolling history at session open; quarantined changes
    additionally suppress all dynamic availability for 1,200 official slots.
    """
    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (close, high, low, volume, amount)
    ]
    if len({len(value) for value in arrays}) != 1 or not len(arrays[0]):
        raise CourageStrictC4FeatureError(
            "stock arrays must be non-empty and equal length"
        )
    close, high, low, volume, amount = arrays
    length = len(close)
    source_valid = (
        np.isfinite(close)
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(volume)
        & np.isfinite(amount)
        & (close > 0)
        & (high >= close)
        & (low <= close)
        & (low > 0)
        & (volume > 0)
        & (amount > 0)
    )
    age = _last_mark_age(
        length, np.asarray(event_offsets, dtype=np.int64), initial_offset=0
    )
    quarantine_age = _last_mark_age(
        length, np.asarray(quarantine_offsets, dtype=np.int64), initial_offset=-1200
    )
    quarantine_block = quarantine_age < 1200
    values = np.zeros((length, len(STOCK_FEATURES)), dtype=np.float32)
    available = np.zeros_like(values, dtype=bool)
    missing = np.zeros_like(values, dtype=bool)

    def assign(
        index: int, raw: np.ndarray, structural: np.ndarray, operand_valid: np.ndarray
    ) -> None:
        finite = np.isfinite(raw)
        structural = np.asarray(structural, dtype=bool) & ~quarantine_block
        valid = structural & operand_valid & finite
        values[valid, index] = raw[valid].astype(np.float32)
        available[:, index] = structural
        missing[:, index] = structural & ~valid

    for index, lag in enumerate((1, 5, 20)):
        raw = np.full(length, np.nan, dtype=np.float64)
        raw[lag:] = close[lag:] / close[:-lag] - 1.0
        operand = np.zeros(length, dtype=bool)
        operand[lag:] = source_valid[lag:] & source_valid[:-lag]
        assign(index, raw, age >= lag, operand)

    valid_count20 = _rolling_sum(source_valid.astype(np.float64), 20)
    sum_amount20 = _rolling_sum(np.where(source_valid, amount, 0.0), 20)
    sum_volume20 = _rolling_sum(np.where(source_valid, volume, 0.0), 20)
    rolling_vwap20 = np.divide(
        sum_amount20,
        sum_volume20,
        out=np.full(length, np.nan, dtype=np.float64),
        where=sum_volume20 > 0,
    )
    assign(
        3,
        close / rolling_vwap20 - 1.0,
        age >= 19,
        source_valid & (valid_count20 == 20),
    )

    ret1 = np.full(length, np.nan, dtype=np.float64)
    ret1[1:] = close[1:] / close[:-1] - 1.0
    ret1_valid = np.zeros(length, dtype=bool)
    ret1_valid[1:] = source_valid[1:] & source_valid[:-1] & (age[1:] >= 1)
    ret_sum = _rolling_sum(np.where(ret1_valid, ret1, 0.0), 20)
    ret_sq_sum = _rolling_sum(np.where(ret1_valid, ret1 * ret1, 0.0), 20)
    ret_count = _rolling_sum(ret1_valid.astype(np.float64), 20)
    variance = np.maximum(ret_sq_sum / 20.0 - (ret_sum / 20.0) ** 2, 0.0)
    assign(4, np.sqrt(variance), age >= 20, ret_count == 20)

    rolling_high = _rolling_max_min(np.where(source_valid, high, np.nan), 20, "max")
    rolling_low = _rolling_max_min(np.where(source_valid, low, np.nan), 20, "min")
    assign(
        5,
        rolling_high / rolling_low - 1.0,
        age >= 19,
        (valid_count20 == 20) & (rolling_low > 0),
    )

    prior_valid = np.concatenate(([False], source_valid[:-1]))
    prior_volume = np.concatenate(
        ([0.0], np.where(source_valid[:-1], volume[:-1], 0.0))
    )
    prior_amount = np.concatenate(
        ([0.0], np.where(source_valid[:-1], amount[:-1], 0.0))
    )
    prior_count20 = _rolling_sum(prior_valid.astype(np.float64), 20)
    prior_volume20 = _rolling_sum(prior_volume, 20) / 20.0
    prior_amount20 = _rolling_sum(prior_amount, 20) / 20.0
    volume_ratio20 = np.divide(
        volume,
        prior_volume20,
        out=np.full(length, np.nan, dtype=np.float64),
        where=prior_volume20 > 0,
    )
    amount_ratio20 = np.divide(
        amount,
        prior_amount20,
        out=np.full(length, np.nan, dtype=np.float64),
        where=prior_amount20 > 0,
    )
    assign(
        6,
        volume_ratio20,
        age >= 20,
        source_valid & (prior_count20 == 20) & (prior_volume20 > 0),
    )
    assign(
        7,
        amount_ratio20,
        age >= 20,
        source_valid & (prior_count20 == 20) & (prior_amount20 > 0),
    )
    if np.isinf(values).any() or np.any(
        available & missing & np.isfinite(values) & (values != 0)
    ):
        raise CourageStrictC4FeatureError("stock feature invariant failed")
    return values, available, missing


def compute_sector_features_day_v1(
    *,
    stock_values: np.ndarray,
    stock_available: np.ndarray,
    stock_missing: np.ndarray,
    sector_codes: np.ndarray,
    active_symbols: np.ndarray,
    minimum_coverage: float = 0.80,
    minimum_valid_count: int = 3,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Aggregate four sector features on a complete daily admitted cross-section."""
    values = np.asarray(stock_values, dtype=np.float32)
    available = np.asarray(stock_available, dtype=bool)
    missing = np.asarray(stock_missing, dtype=bool)
    codes = np.asarray(sector_codes, dtype=object)
    active = np.asarray(active_symbols, dtype=bool)
    if values.ndim != 3 or values.shape[1:] != (240, len(STOCK_FEATURES)):
        raise CourageStrictC4FeatureError("stock day tensor must be [symbols,240,8]")
    if available.shape != values.shape or missing.shape != values.shape:
        raise CourageStrictC4FeatureError("stock day masks drift")
    if len(codes) != len(values) or len(active) != len(values):
        raise CourageStrictC4FeatureError("sector identity length drift")
    if not 0 < minimum_coverage <= 1 or minimum_valid_count < 1:
        raise CourageStrictC4FeatureError("invalid sector coverage gate")
    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    known_codes = sorted(
        {str(code) for code in codes[active] if pd.notna(code) and str(code)}
    )
    for code in known_codes:
        members = active & (codes.astype(str) == code)
        denominator = int(members.sum())
        output = np.zeros((240, len(SECTOR_FEATURES)), dtype=np.float32)
        output_available = np.zeros_like(output, dtype=bool)
        output_missing = np.zeros_like(output, dtype=bool)
        counts = np.zeros_like(output, dtype=np.int16)
        for output_index, feature in enumerate(SECTOR_FEATURES):
            source_index = SECTOR_SOURCE_INDEX[feature]
            valid = (
                available[members, :, source_index]
                & ~missing[members, :, source_index]
                & np.isfinite(values[members, :, source_index])
            )
            structural_count = available[members, :, source_index].sum(axis=0)
            structural_gate = (structural_count >= minimum_valid_count) & (
                structural_count / denominator >= minimum_coverage
            )
            counts[:, output_index] = valid.sum(axis=0).astype(np.int16)
            coverage_valid = (counts[:, output_index] >= minimum_valid_count) & (
                counts[:, output_index] / denominator >= minimum_coverage
            )
            source = np.where(valid, values[members, :, source_index], np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if feature == "sector_positive_breadth_1":
                    raw = np.nanmean(np.where(valid, source > 0, np.nan), axis=0)
                else:
                    raw = np.nanmedian(source, axis=0)
            finite = np.isfinite(raw)
            accepted = coverage_valid & finite
            output[accepted, output_index] = raw[accepted].astype(np.float32)
            output_available[:, output_index] = structural_gate
            output_missing[:, output_index] = structural_gate & ~accepted
        result[code] = (output, output_available, output_missing, counts)
    return result


def build_slow_context_v1(
    *, membership: pd.DataFrame, daily: pd.DataFrame, official_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Build the five slow values using only the previous completed session."""
    required_membership = {
        "instrument",
        "signal_date",
        "turnover_mean_60_percent",
        "strict_T_minus_1",
    }
    required_daily = {
        "instrument",
        "session_date",
        "close_cny",
        "total_mv_10k_cny",
        "strict_t_minus_1_source_use_authorized",
        "same_session_intraday_use_authorized",
    }
    if not required_membership.issubset(membership) or not required_daily.issubset(
        daily
    ):
        raise CourageStrictC4FeatureError("slow-context input schema drift")
    dates = (
        pd.DatetimeIndex(pd.to_datetime(official_dates))
        .normalize()
        .unique()
        .sort_values()
    )
    source = daily.copy()
    source["session_date"] = pd.to_datetime(source["session_date"]).dt.normalize()
    source = source.set_index(["instrument", "session_date"]).sort_index()
    output = membership.loc[
        :, ["instrument", "signal_date", "turnover_mean_60_percent", "strict_T_minus_1"]
    ].copy()
    output["signal_date"] = pd.to_datetime(output["signal_date"]).dt.normalize()
    rows: list[dict[str, object]] = []
    for row in output.itertuples(index=False):
        signal_date = pd.Timestamp(row.signal_date)
        # The slow context for the first signal session is built from history
        # ending at T-1.  The historical calendar input therefore need not
        # contain T itself; it only needs to contain the completed sessions.
        position = int(dates.searchsorted(signal_date, side="left"))
        prior_dates = dates[:position]
        if len(prior_dates) == 0:
            raise CourageStrictC4FeatureError(
                "slow context has no prior official session"
            )
        instrument_daily = (
            source.loc[str(row.instrument)]
            if str(row.instrument) in source.index.levels[0]
            else pd.DataFrame()
        )
        closes = (
            pd.to_numeric(instrument_daily["close_cny"], errors="coerce")
            .reindex(prior_dates)
            .to_numpy(dtype=np.float64)
            if not instrument_daily.empty
            else np.full(len(prior_dates), np.nan)
        )
        latest = (
            instrument_daily.reindex(prior_dates).iloc[-1] if len(prior_dates) else None
        )
        values = np.full(len(SLOW_FEATURES), np.nan, dtype=np.float64)
        if (
            len(closes) >= 6
            and np.isfinite(closes[-1])
            and np.isfinite(closes[-6])
            and closes[-6] > 0
        ):
            values[0] = closes[-1] / closes[-6] - 1.0
        if (
            len(closes) >= 20
            and np.isfinite(closes[-20:]).all()
            and closes[-20:].mean() > 0
        ):
            values[1] = closes[-1] / closes[-20:].mean() - 1.0
        if (
            len(closes) >= 21
            and np.isfinite(closes[-21:]).all()
            and (closes[-21:-1] > 0).all()
        ):
            returns = closes[-20:] / closes[-21:-1] - 1.0
            values[2] = returns.std(ddof=0)
        values[3] = float(row.turnover_mean_60_percent)
        if latest is not None:
            authorized = bool(
                latest["strict_t_minus_1_source_use_authorized"]
            ) and not bool(latest["same_session_intraday_use_authorized"])
            market_cap = float(latest["total_mv_10k_cny"])
            if authorized and np.isfinite(market_cap) and market_cap > 0:
                values[4] = np.log(market_cap)
        record: dict[str, object] = {
            "instrument": str(row.instrument),
            "signal_date": pd.Timestamp(row.signal_date),
            "slow_source_date": prior_dates[-1],
        }
        for index, feature in enumerate(SLOW_FEATURES):
            valid = bool(np.isfinite(values[index]))
            record[feature] = float(values[index]) if valid else 0.0
            record[f"{feature}__available"] = True
            record[f"{feature}__data_missing"] = not valid
        rows.append(record)
    result = pd.DataFrame(rows)
    if result.duplicated(["instrument", "signal_date"]).any():
        raise CourageStrictC4FeatureError("slow-context key duplication")
    if not (result["slow_source_date"] < result["signal_date"]).all():
        raise CourageStrictC4FeatureError("slow-context T-1 violation")
    return result

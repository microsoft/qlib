# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Data-provider protocol decoupling the backtest engine from ``qlib.data``.

``Exchange`` historically calls ``D.features(...)`` / ``D.instruments(...)``
directly, which forces every order-matching unit test to provision a local
dataset. This module defines the minimal :class:`QuoteProvider` protocol plus
two implementations:

- :class:`QlibDataProvider` — thin adapter over the live ``qlib.data`` layer
  (production path, unchanged behavior);
- :class:`InMemoryQuoteProvider` — pure in-memory stub for filesystem-free
  domain-logic tests (no ``qlib.init()``, no ``~/.qlib`` data).

``Exchange`` does not depend on this module yet (backward compatibility); new
execution-engine code should accept a ``QuoteProvider`` and be tested with the
in-memory implementation.
"""

from __future__ import annotations

from typing import Any, List, Protocol, Union, runtime_checkable

import pandas as pd


@runtime_checkable
class QuoteProvider(Protocol):
    """Minimal market-data surface required by the execution engine."""

    def instruments(self, market: Union[list, str]) -> list:
        """Resolve a market name (e.g. ``csi300``) to a list of instrument codes."""
        ...  # pragma: no cover - protocol stub

    def features(
        self,
        instruments: Union[list, str],
        fields: List[str],
        start_time: Any = None,
        end_time: Any = None,
        freq: str = "day",
    ) -> pd.DataFrame:
        """Return a ``quote_df``-shaped frame for the requested window."""
        ...  # pragma: no cover - protocol stub


class QlibDataProvider:
    """Production adapter delegating to ``qlib.data.data.D`` lazily."""

    def instruments(self, market: Union[list, str]) -> list:
        from qlib.data.data import D

        return D.instruments(market)  # type: ignore[no-any-return]

    def features(
        self,
        instruments: Union[list, str],
        fields: List[str],
        start_time: Any = None,
        end_time: Any = None,
        freq: str = "day",
    ) -> pd.DataFrame:
        from qlib.data.data import D

        return D.features(instruments, fields, start_time, end_time, freq=freq)  # type: ignore[no-any-return]


class InMemoryQuoteProvider:
    """In-memory stub: tests inject one pre-built ``quote_df`` frame."""

    def __init__(self, quote_df: pd.DataFrame | None = None, codes: list | None = None) -> None:
        self._quote_df = quote_df
        self._codes = list(codes) if codes is not None else None

    def instruments(self, market: Union[list, str]) -> list:
        if isinstance(market, list):
            return market
        if self._codes is not None:
            return list(self._codes)
        if self._quote_df is not None:
            try:
                return list(self._quote_df.index.get_level_values(0).unique())
            except (AttributeError, IndexError, KeyError):
                return []
        return []

    def features(
        self,
        instruments: Union[list, str],
        fields: List[str],
        start_time: Any = None,
        end_time: Any = None,
        freq: str = "day",
    ) -> pd.DataFrame:
        if self._quote_df is None:
            raise ValueError("InMemoryQuoteProvider has no quote_df injected")
        frame = self._quote_df
        missing = [field for field in fields if field not in frame.columns]
        if missing:
            raise ValueError(f"quote frame is missing fields: {missing}")
        if isinstance(instruments, list):
            try:
                frame = frame.loc[instruments]
            except KeyError as exc:
                raise ValueError(f"unknown instruments: {instruments}") from exc
        return frame


__all__ = ["InMemoryQuoteProvider", "QlibDataProvider", "QuoteProvider"]

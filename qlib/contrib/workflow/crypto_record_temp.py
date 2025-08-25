# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Crypto-specific portfolio analysis record.

This module provides `CryptoPortAnaRecord`, a non-intrusive extension of
`qlib.workflow.record_temp.PortAnaRecord` that adapts portfolio analysis for
crypto markets (e.g., 365-day annualization, product compounding) while keeping
the default Qlib behavior unchanged for other users.
"""

from __future__ import annotations

from typing import List, Union

import pandas as pd

from ..evaluate import risk_analysis as original_risk_analysis
from ...utils import fill_placeholder, get_date_by_shift
from ...workflow.record_temp import PortAnaRecord


def _crypto_risk_analysis(r: pd.Series, N: int = 365) -> pd.Series:
    """Risk analysis with product compounding and 365 annual days.

    This wraps Qlib's contrib risk_analysis with crypto-friendly defaults by
    passing N and forcing product mode through freq=None.
    """
    return original_risk_analysis(r, freq=None, N=N, mode="product")


class CryptoPortAnaRecord(PortAnaRecord):
    """A crypto-friendly PortAnaRecord.

    Differences vs PortAnaRecord (only when used):
    - Annualization uses 365 trading days.
    - Product compounding for cumulative/excess returns.
    - Optionally align exchange freq based on risk_analysis_freq if provided.

    Defaults and behavior of the core PortAnaRecord remain unchanged elsewhere.
    """

    def __init__(
        self,
        recorder,
        config=None,
        risk_analysis_freq: Union[List, str] = None,
        indicator_analysis_freq: Union[List, str] = None,
        indicator_analysis_method=None,
        crypto_annual_days: int = 365,
        skip_existing: bool = False,
        **kwargs,
    ):
        super().__init__(
            recorder=recorder,
            config=config,
            risk_analysis_freq=risk_analysis_freq,
            indicator_analysis_freq=indicator_analysis_freq,
            indicator_analysis_method=indicator_analysis_method,
            skip_existing=skip_existing,
            **kwargs,
        )
        self.crypto_annual_days = crypto_annual_days

    def _generate(self, **kwargs):  # override only the generation logic
        from ...backtest import backtest as normal_backtest

        pred = self.load("pred.pkl")

        # Replace placeholder values
        placeholder_value = {"<PRED>": pred}
        for k in "executor_config", "strategy_config":
            setattr(self, k, fill_placeholder(getattr(self, k), placeholder_value))

        # Auto-extract time range if not set
        dt_values = pred.index.get_level_values("datetime")
        if self.backtest_config["start_time"] is None:
            self.backtest_config["start_time"] = dt_values.min()
        if self.backtest_config["end_time"] is None:
            self.backtest_config["end_time"] = get_date_by_shift(dt_values.max(), 1)

        # Optionally align exchange frequency with requested risk analysis frequency
        try:
            target_freq = None
            raf = getattr(self, "risk_analysis_freq", None)
            if isinstance(raf, (list, tuple)) and len(raf) > 0:
                target_freq = raf[0]
            elif isinstance(raf, str):
                target_freq = raf
            if isinstance(target_freq, str) and target_freq:
                ex_kwargs = dict(self.backtest_config.get("exchange_kwargs", {}) or {})
                ex_kwargs.setdefault("freq", target_freq)
                self.backtest_config["exchange_kwargs"] = ex_kwargs
        except Exception:
            pass

        # Run backtest
        portfolio_metric_dict, indicator_dict = normal_backtest(
            executor=self.executor_config, strategy=self.strategy_config, **self.backtest_config
        )

        artifact_objects = {}

        # Save portfolio metrics; also attach crypto metrics as attrs for consumers
        for _freq, (report_normal, positions_normal) in portfolio_metric_dict.items():
            if "return" in report_normal.columns:
                r = report_normal["return"].astype(float).fillna(0)
                b = report_normal["bench"].astype(float).fillna(0)
                c = report_normal.get("cost", 0.0)
                c = c.astype(float).fillna(0) if isinstance(c, pd.Series) else float(c)

                # Product compounding cum NAVs
                nav_b = (1 + b).cumprod()
                nav_s0 = (1 + r).cumprod()
                nav_s1 = (1 + (r - c)).cumprod()

                # Attach crypto metrics for downstream use (non-breaking)
                try:
                    report_normal.attrs["crypto_metrics"] = {
                        "strategy": _crypto_risk_analysis(r, N=self.crypto_annual_days),
                        "benchmark": _crypto_risk_analysis(b, N=self.crypto_annual_days),
                        "excess_wo_cost": _crypto_risk_analysis((1 + r) / (1 + b) - 1, N=self.crypto_annual_days),
                        "excess_w_cost": _crypto_risk_analysis((1 + (r - c)) / (1 + b) - 1, N=self.crypto_annual_days),
                        "annual_days": self.crypto_annual_days,
                    }
                except Exception:
                    pass

            artifact_objects.update({f"report_normal_{_freq}.pkl": report_normal})
            artifact_objects.update({f"positions_normal_{_freq}.pkl": positions_normal})

        for _freq, indicators_normal in indicator_dict.items():
            artifact_objects.update({f"indicators_normal_{_freq}.pkl": indicators_normal[0]})
            artifact_objects.update({f"indicators_normal_{_freq}_obj.pkl": indicators_normal[1]})

        # Risk analysis (365 days, product mode) printing and artifacts, mirroring PortAnaRecord
        for _analysis_freq in self.risk_analysis_freq:
            if _analysis_freq not in portfolio_metric_dict:
                import warnings

                warnings.warn(
                    f"the freq {_analysis_freq} report is not found, please set the corresponding env with `generate_portfolio_metrics=True`"
                )
            else:
                report_normal, _ = portfolio_metric_dict.get(_analysis_freq)
                analysis = dict()

                r = report_normal["return"].astype(float).fillna(0)
                b = report_normal["bench"].astype(float).fillna(0)
                c = report_normal.get("cost", 0.0)
                c = c.astype(float).fillna(0) if isinstance(c, pd.Series) else float(c)

                # geometric excess
                analysis["excess_return_without_cost"] = _crypto_risk_analysis((1 + r) / (1 + b) - 1, N=self.crypto_annual_days)
                analysis["excess_return_with_cost"] = _crypto_risk_analysis((1 + (r - c)) / (1 + b) - 1, N=self.crypto_annual_days)

                analysis_df = pd.concat(analysis)
                from ...utils import flatten_dict

                analysis_dict = flatten_dict(analysis_df["risk"].unstack().T.to_dict())
                self.recorder.log_metrics(**{f"{_analysis_freq}.{k}": v for k, v in analysis_dict.items()})
                artifact_objects.update({f"port_analysis_{_analysis_freq}.pkl": analysis_df})

        return artifact_objects


__all__ = ["CryptoPortAnaRecord"]



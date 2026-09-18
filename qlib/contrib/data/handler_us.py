# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
US stock data handlers with fundamental factors.

These handlers extend Alpha158 with fundamental factors collected from
Yahoo Finance + SEC EDGAR filing dates (Route 1.5 approach).

The fundamental factors are stored as Qlib features (bin files) alongside
the standard OHLCV data. They are expected to be pre-computed and forward-
filled to daily frequency before being dumped to Qlib format.

Available handlers:
    - USAlpha158: Alpha158 technical factors only, tuned for US market
    - USFundamental: Fundamental factors only
    - USAlphaFundamental: Combined technical + fundamental factors (recommended)

Usage:
    See examples/us_fundamental/workflow_config.yaml for a complete example.
"""

from qlib.contrib.data.handler import Alpha158, _DEFAULT_LEARN_PROCESSORS, _DEFAULT_INFER_PROCESSORS, check_transform_proc
from qlib.contrib.data.loader import Alpha158DL
from qlib.data.dataset.handler import DataHandlerLP


# ── Fundamental factor features ──────────────────────────────────────────────
# These correspond to bin files produced by build_factors.py:
#   features/<SYMBOL>/roe.day.bin, features/<SYMBOL>/roa.day.bin, etc.

FUNDAMENTAL_FIELDS = [
    # Quality factors
    "$roe",               # Return on Equity (quarterly, forward-filled)
    "$roa",               # Return on Assets
    "$gross_margin",      # Gross Profit / Revenue
    "$accruals",          # (NetIncome - OperatingCashFlow) / TotalAssets

    # Leverage
    "$debt_to_equity",    # TotalDebt / StockholdersEquity

    # Growth factors (YOY)
    "$revenue_yoy",       # Revenue growth vs same quarter last year
    "$earnings_yoy",      # Earnings growth vs same quarter last year
]

FUNDAMENTAL_NAMES = [
    "ROE", "ROA", "GMARGIN", "ACCRUALS",
    "DE_RATIO",
    "REV_YOY", "EARN_YOY",
]

# Price-relative factors (need to be divided by market cap or price)
# These use raw fundamental values from bin files + current price
PRICE_RELATIVE_FIELDS = [
    # EP = NetIncome / (Close * SharesOutstanding) ≈ NetIncome / MarketCap
    # Since we don't have shares outstanding in daily data, we use the
    # pre-computed quarterly NetIncome and normalize by close price.
    # This gives a "per-dollar-of-price" measure, comparable across stocks
    # within the cross-sectional normalization.
    "$netincome / ($close + 1e-12)",         # Earnings yield proxy
    "$totalrevenue / ($close + 1e-12)",      # Sales yield proxy
    "$freecashflow / ($close + 1e-12)",      # FCF yield proxy
    "$stockholdersequity / ($close + 1e-12)", # Book yield proxy
    "$ebitda / ($close + 1e-12)",            # EBITDA yield proxy
]

PRICE_RELATIVE_NAMES = [
    "EARN_YIELD", "SALES_YIELD", "FCF_YIELD", "BOOK_YIELD", "EBITDA_YIELD",
]


class USAlpha158(DataHandlerLP):
    """Alpha158 technical factors tuned for US stocks.

    Changes from standard Alpha158:
        - Extended rolling windows (up to 250 days) for momentum
        - Added 12-1 month momentum factor (academically proven for US)
        - Added overnight gap factor (no price limits in US market)
        - Removed VWAP from price features (often unavailable in free data)
    """

    def __init__(
        self,
        instruments="sp500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_feature_config(self):
        # Alpha158 with US-tuned config
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW"],  # No VWAP
            },
            "rolling": {
                "windows": [5, 10, 20, 30, 60, 120, 250],  # Extended windows
                "exclude": ["RANK"],
            },
        }
        fields, names = Alpha158DL.get_feature_config(conf)

        # Add US-specific technical factors
        extra_fields = [
            # 12-1 month momentum (Jegadeesh & Titman)
            "Ref($close, 250)/$close - Ref($close, 20)/$close",
            # Annualized volatility
            "Std($close/Ref($close,1)-1, 250)",
            # Volume surge (relative to long-term average)
            "Mean($volume, 5) / (Mean($volume, 120) + 1e-12)",
            # Overnight gap (US market has no price limits)
            "$open / Ref($close, 1) - 1",
            # Intraday range trend
            "Mean(($high-$low)/$open, 20) / (Mean(($high-$low)/$open, 120) + 1e-12)",
        ]
        extra_names = [
            "MOM_12_1", "VOL_250", "VOLUME_SURGE", "GAP", "RANGE_TREND",
        ]

        return fields + extra_fields, names + extra_names

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]


class USFundamental(DataHandlerLP):
    """Fundamental-only factors for US stocks.

    Uses pre-computed fundamental factors stored as Qlib features.
    Requires running the us_fundamental data collector pipeline first.
    """

    def __init__(
        self,
        instruments="sp500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_feature_config(self):
        fields = FUNDAMENTAL_FIELDS + PRICE_RELATIVE_FIELDS
        names = FUNDAMENTAL_NAMES + PRICE_RELATIVE_NAMES
        return fields, names

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]


class USAlphaFundamental(DataHandlerLP):
    """Combined technical (Alpha158) + fundamental factors for US stocks.

    This is the recommended handler for US stock prediction. It combines:
        - Alpha158 technical factors (tuned for US market)
        - Fundamental quality/value/growth factors
        - Price-relative fundamental factors

    Total: ~180 features (158 tech + ~12 US-specific tech + ~12 fundamental)
    """

    def __init__(
        self,
        instruments="sp500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_feature_config(self):
        # Start with US-tuned Alpha158 technical factors
        tech_handler = USAlpha158.__new__(USAlpha158)
        tech_fields, tech_names = tech_handler.get_feature_config()

        # Add fundamental factors
        fund_fields = FUNDAMENTAL_FIELDS + PRICE_RELATIVE_FIELDS
        fund_names = FUNDAMENTAL_NAMES + PRICE_RELATIVE_NAMES

        return tech_fields + fund_fields, tech_names + fund_names

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]

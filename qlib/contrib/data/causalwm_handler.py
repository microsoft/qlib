# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""CausalWM357 — 357 weight-free, closed-form world-model features.

``CausalWM357`` is an ``Alpha158``/``Alpha360`` sibling built on a *discovered*
feature set: 111 closed-form programs producing 357 columns, found by an
explanatory search (does the candidate improve prediction of the market's own
next state?) rather than by fitting returns.  **There are no learned weights
anywhere** — every column is a frozen closed form, so the handler re-executes
rather than re-fits, and every column is auditable: ``ret_weighted_reversal_0``
is produced by the function ``ret_weighted_reversal``, printable with
``python -m causalwm --show ret_weighted_reversal``.

The feature math lives in the optional ``causalwm`` package (pip installable,
MIT, pure PyTorch used as an array library); this module is only the Qlib
adapter, so qlib carries no new numerics::

    pip install causalwm

How it differs from Alpha158 / Alpha360
---------------------------------------
* **Weekly rows.**  Daily bars are pulled through Qlib's data API and
  resampled internally to W-FRI weekly bars (first/max/min/last/sum); the
  handler serves one row per instrument-week, indexed by the week-end date.
  This makes no claim on qlib ``freq="week"`` stores.
* **Cross-sectional.**  Alpha158's expressions are per-instrument.  These
  features are built from per-week cross-sectional Gaussianized ranks over the
  loaded universe, so a value depends on which instruments were loaded, and
  ``inst_processors`` (per-instrument processing) is refused rather than
  silently changing their meaning.
* **Full-history universe.**  The encoder needs a gapless 52-week history per
  instrument; instruments entering or leaving mid-span are dropped, with a
  warning reporting the kept count.
* **Two-week signal.**  This is a weekly mechanism, not a daily alpha; see the
  PR description for measured numbers at both horizons.
* **Not expressible in the Qlib DSL.**  Cross-sectional Gaussianized ranks and
  the closed-form programs over 52-week windows have no operator-language
  form, so the features are computed by a small Python ``DataLoader`` instead
  of ``QlibDataLoader``.

Causality and the availability lag
----------------------------------
The phi row computed from weekly bars through Friday *W* is stamped at Friday
``W + availability_lag_weeks``.  The default lag is **1 week**: the row served
at Friday *W* was already fully computable at the close of Friday *W-1*, so it
needs no same-bar data to be traded on at *W*'s close.  ``0`` restores the
source paper's protocol.  Either way ``feature[t]`` depends only on raw data at
or before *t* — the channel constructions are strictly past-only (expanding
demeans, trailing windows, contemporaneous cross-sectional ranks).  The lag
counts weekly *bars*: a week in which the whole market was closed has no bar,
so a lagged stamp can land more than seven calendar days later (never fewer).

Label
-----
The default is Alpha158's convention, ``Ref($close, -2)/Ref($close, -1) - 1``,
evaluated **on the weekly grid** (buy at next week's close, sell the week
after).  Any ``Ref($close, -a)/Ref($close, -b) - 1`` is accepted (``$close``
standing in for ``Ref($close, 0)``), so the two-week horizon the feature set
was discovered at is ``label=(["Ref($close, -2)/$close - 1"], ["LABEL0"])``.
Because this loader builds weekly bars in Python it does not run Qlib's
expression engine; any other expression is rejected with an explanatory
message rather than silently mis-evaluated.

Features are served RAW; the default ``infer_processors`` normalize them with a
fit-ranged ``RobustZScoreNorm`` (so ``fit_start_time``/``fit_end_time`` are
required, as for ``Alpha360``).

Example
-------
.. code-block:: yaml

    handler:
        class: CausalWM357
        module_path: qlib.contrib.data.causalwm_handler
        kwargs:
            start_time: 2016-01-01
            end_time: 2020-08-01
            fit_start_time: 2017-01-01
            fit_end_time: 2018-12-31
            instruments: csi300
"""
import pandas as pd

from qlib.data import D

from .handler import check_transform_proc
from ...data.dataset.handler import DataHandlerLP
from ...data.dataset.loader import DataLoader

_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]
_DEFAULT_INFER_PROCESSORS = [
    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
]

#: Alpha158's label convention, evaluated on the weekly grid.
DEFAULT_LABEL = (["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"])

_IMPORT_HINT = (
    "CausalWM357 needs the optional `causalwm` package (which brings PyTorch; CPU is fine):\n"
    "    pip install causalwm\n"
    "It ships the 111 closed-form feature programs; they carry ZERO learned weights, "
    "torch is used only as an array library for the forward pass."
)


def _causalwm():
    """Import the optional dependency lazily, with an actionable message.

    torch is not a qlib dependency (though 33 modules under ``qlib/contrib``
    already import it), so nothing here is imported until features or feature
    names are actually requested.
    """
    try:
        from causalwm import qlib_handler as _impl
    except ImportError as exc:
        raise ImportError(_IMPORT_HINT) from exc
    return _impl


class CausalWM357DL(DataLoader):
    """Compute the CausalWM357 feature group in Python.

    Configured by dict so the owning handler stays picklable; holds no torch
    state — the encoder is built inside :meth:`load` and dropped again.
    """

    def __init__(
        self,
        label=DEFAULT_LABEL,
        freq: str = "day",
        availability_lag_weeks: int = 1,
        week_anchor: str = "W-FRI",
        warmup_days: int = 420,
        filter_pipe=None,
    ):
        self.label = label
        self.freq = freq
        self.availability_lag_weeks = availability_lag_weeks
        self.week_anchor = week_anchor
        self.warmup_days = warmup_days  # >= 52 weeks of window + holiday slack
        self.filter_pipe = filter_pipe

    def load(self, instruments=None, start_time=None, end_time=None) -> pd.DataFrame:
        if self.freq != "day":
            raise ValueError(
                "CausalWM357DL reads daily bars and resamples them to weekly bars "
                f"internally; freq={self.freq!r} is not supported"
            )
        impl = _causalwm()
        if isinstance(instruments, str):
            instruments = D.instruments(market=instruments, filter_pipe=self.filter_pipe)
        ext_start = (
            None
            if start_time is None
            else pd.Timestamp(start_time) - pd.Timedelta(days=self.warmup_days)
        )
        daily = D.features(
            instruments,
            ["$open", "$high", "$low", "$close", "$volume"],
            start_time=ext_start,
            end_time=end_time,
            freq="day",
        ).reset_index()
        daily.columns = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        df = impl.compute_weekly_features(
            daily,
            label=self.label,
            availability_lag_weeks=self.availability_lag_weeks,
            week_anchor=self.week_anchor,
            core_only=True,
        )
        return df.loc[pd.IndexSlice[slice(start_time, end_time), :], :]


class CausalWM357(DataHandlerLP):
    """357 discovered, weight-free, closed-form features (111 programs).

    See the module docstring for the weekly grid, the causal availability lag,
    the label convention and the cross-sectional caveats.
    """

    N_PROGRAMS = 111
    N_FEATURES = 357

    def __init__(
        self,
        instruments="csi300",
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
        availability_lag_weeks=1,
        week_anchor="W-FRI",
        warmup_days=420,
        **kwargs,
    ):
        if inst_processors:
            raise NotImplementedError(
                "CausalWM357 features are cross-sectional over the loaded universe, "
                "so per-instrument processors would change their meaning; leave "
                "inst_processors empty"
            )
        label = kwargs.pop("label", self.get_label_config())
        if label is not None:  # fail fast, before any data is touched
            for expr in label[0]:
                _causalwm()._parse_close_ratio_label(expr)
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "CausalWM357DL",
            "module_path": __name__,
            "kwargs": {
                "label": label,
                "freq": freq,
                "availability_lag_weeks": availability_lag_weeks,
                "week_anchor": week_anchor,
                "warmup_days": warmup_days,
                "filter_pipe": filter_pipe,
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

    @staticmethod
    def get_feature_config():
        """The 357 served column names, in phi order (``f"{program}_{k}"``)."""
        return _causalwm().core_feature_names()

    def get_label_config(self):
        return DEFAULT_LABEL

import os
import sys

import pytest

"""Ignore RL tests on non-linux platform."""
collect_ignore = []

if sys.platform != "linux":
    for root, dirs, files in os.walk("rl"):
        for file in files:
            collect_ignore.append(os.path.join(root, file))


@pytest.fixture
def workflow_context(tmp_path, monkeypatch, request):
    """An offline daily market and isolated recorder store for real workflows."""
    from copy import deepcopy
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd
    import qlib
    from qlib.config import C
    from qlib.data.cache import H
    from qlib.workflow import R

    calendar = pd.bdate_range("2020-01-01", periods=260)
    instruments = [f"SH{600000 + i:06d}" for i in range(getattr(request, "param", 8))]
    provider_uri = tmp_path / "market"
    calendars = provider_uri / "calendars"
    calendars.mkdir(parents=True)
    calendars.joinpath("day.txt").write_text("\n".join(calendar.strftime("%Y-%m-%d")) + "\n")
    instruments_dir = provider_uri / "instruments"
    instruments_dir.mkdir()
    spans = "".join(f"{symbol}\t{calendar[0]:%Y-%m-%d}\t{calendar[-1]:%Y-%m-%d}\n" for symbol in instruments)
    instruments_dir.joinpath("csi300.txt").write_text(spans)
    instruments_dir.joinpath("all.txt").write_text(spans)
    random = np.random.RandomState(42)
    market_return = random.normal(0.0004, 0.008, len(calendar))
    for i, symbol in enumerate(instruments + ["SH000300"]):
        stock_return = random.normal(0, 0.006, len(calendar))
        for day in range(1, len(calendar)):
            stock_return[day] += 0.7 * stock_return[day - 1]
        returns = market_return + stock_return
        close = (20 + i) * np.exp(np.cumsum(returns))
        opening = close * (1 + random.normal(0, 0.003, len(calendar)))
        values = {
            "close": close,
            "open": opening,
            "high": np.maximum(close, opening) * 1.01,
            "low": np.minimum(close, opening) * 0.99,
            "vwap": (close + opening) / 2,
            "volume": random.uniform(1000000, 2000000, len(calendar)),
            "factor": np.ones(len(calendar)),
        }
        directory = provider_uri / "features" / symbol.lower()
        directory.mkdir(parents=True)
        for field, array in values.items():
            np.concatenate(([0], array)).astype("<f4").tofile(directory / f"{field}.day.bin")

    previous = deepcopy(C)
    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    qlib.init(
        provider_uri=str(provider_uri),
        kernels=1,
        expression_cache=None,
        dataset_cache=None,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {"uri": uri, "default_exp_name": "workflow"},
        },
    )
    try:
        yield SimpleNamespace(
            root=tmp_path, provider_uri=provider_uri, uri=uri, calendar=calendar, instruments=instruments
        )
    finally:
        R.end_exp()
        H.clear()
        C.set_conf_from_C(previous)
        if previous.registered:
            C.register()

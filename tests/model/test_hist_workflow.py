# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
from pathlib import Path
import site
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


def _init_workflow(context):
    import qlib

    torch.set_num_threads(1)
    np.random.seed(42)
    torch.random.default_generator.manual_seed(42)
    qlib.init(
        provider_uri=context["provider_uri"],
        kernels=1,
        expression_cache=None,
        dataset_cache=None,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {"uri": context["uri"], "default_exp_name": "hist-workflow"},
        },
    )


def _train_workflow(context):
    from qlib.contrib.model.pytorch_gru import GRUModel
    from qlib.contrib.model.pytorch_hist import UNKNOWN_STOCK_INDEX
    from qlib.model.trainer import task_train

    root = Path(context["root"])
    concepts = np.zeros((UNKNOWN_STOCK_INDEX + 1, 3), dtype=np.float32)
    for index in range(len(context["instruments"])):
        concepts[index, index % 3] = 1
        concepts[index, (index + 1) % 3] = 1
    np.save(root / "concepts.npy", concepts, allow_pickle=False)
    (root / "stock_index.json").write_text(
        json.dumps({symbol: index for index, symbol in enumerate(context["instruments"])}),
        encoding="utf-8",
    )
    # HIST constructs its pretrained base with these default dimensions.
    torch.save(GRUModel().state_dict(), root / "base_gru.pt")
    segments = context["segments"]
    task = {
        "model": {
            "class": "HIST",
            "module_path": "qlib.contrib.model.pytorch_hist",
            "kwargs": {
                "d_feat": 6,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.0,
                "n_epochs": 1,
                "lr": 0.001,
                "early_stop": 1,
                "metric": "ic",
                "base_model": "GRU",
                "model_path": str(root / "base_gru.pt"),
                "stock2concept": str(root / "concepts.npy"),
                "stock_index": str(root / "stock_index.json"),
                "GPU": -1,
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha360",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": {
                        "instruments": "csi300",
                        "start_time": segments["train"][0],
                        "end_time": segments["test"][1],
                        "fit_start_time": segments["train"][0],
                        "fit_end_time": segments["train"][1],
                        "infer_processors": [
                            {
                                "class": "RobustZScoreNorm",
                                "kwargs": {"fields_group": "feature", "clip_outlier": True},
                            },
                            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                        ],
                        "learn_processors": [
                            {"class": "DropnaLabel"},
                            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
                        ],
                    },
                },
                "segments": segments,
            },
        },
        "record": [
            {"class": "SignalRecord"},
            {"class": "SigAnaRecord", "kwargs": {"ana_long_short": False, "ann_scaler": 252}},
            {
                "class": "PortAnaRecord",
                "kwargs": {
                    "config": {
                        "strategy": {
                            "class": "TopkDropoutStrategy",
                            "module_path": "qlib.contrib.strategy",
                            "kwargs": {"signal": "<PRED>", "topk": 3, "n_drop": 1},
                        },
                        "backtest": {
                            "start_time": context["backtest_dates"][0],
                            "end_time": context["backtest_dates"][-1],
                            "account": 1000000,
                            "benchmark": "SH000300",
                            "exchange_kwargs": {
                                "limit_threshold": 0.095,
                                "deal_price": "close",
                                "open_cost": 0.0005,
                                "close_cost": 0.0015,
                                "min_cost": 5,
                            },
                        },
                    }
                },
            },
        ],
    }
    recorder = task_train(task, "hist-workflow")
    return recorder.id, recorder.experiment_id


def _check_artifacts(context, recorder):
    from qlib.contrib.data.handler import Alpha360
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.utils.exceptions import LoadObjectError
    from qlib.workflow.online.update import RMDLoader
    from qlib.workflow.record_temp import SignalRecord
    from qlib.workflow.recorder import UnsafeArtifactWarning

    assert recorder.status == "FINISHED"
    assert recorder.load_object("task")["model"]["kwargs"]["n_epochs"] == 1
    segments = context["segments"]
    for name, class_name in (("params.pkl", "HIST"), ("dataset", "DatasetH")):
        with pytest.raises(LoadObjectError, match=class_name):
            recorder.load_object(name)
    restricted = RMDLoader(recorder)
    with pytest.raises(LoadObjectError, match="HIST"):
        restricted.get_model()
    with pytest.raises(LoadObjectError, match="DatasetH"):
        restricted.get_dataset(*segments["test"])

    loader = RMDLoader(recorder, trusted_artifacts=True)
    with pytest.warns(UnsafeArtifactWarning):
        model = loader.get_model()
    with pytest.warns(UnsafeArtifactWarning):
        dataset = loader.get_dataset(segments["train"][0], segments["test"][1], segments=segments)
    assert type(model).__name__ == "HIST"
    assert isinstance(dataset, DatasetH)
    assert isinstance(dataset.handler, Alpha360)
    assert model.fitted and model.n_epochs == 1
    assert model.device == torch.device("cpu")
    assert all(parameter.device.type == "cpu" for parameter in model.HIST_model.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in model.HIST_model.parameters())

    for segment, days in (("train", 12), ("valid", 4), ("test", 6)):
        data = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        expected_index = pd.MultiIndex.from_product(
            [pd.bdate_range(*segments[segment]), context["instruments"]], names=["datetime", "instrument"]
        )
        pd.testing.assert_index_equal(data.index, expected_index)
        assert data["feature"].shape == (days * 8, 360)
        assert data["label"].shape == (days * 8, 1)
        assert np.isfinite(data.to_numpy()).all()

    optimizer_state = model.train_optimizer.state_dict()["state"]
    assert optimizer_state
    # Every used parameter must have completed all 12 daily batches of the single epoch.
    assert {int(state["step"]) for state in optimizer_state.values()} == {12}
    initial = torch.load(Path(context["root"]) / "base_gru.pt", map_location="cpu", weights_only=True)
    weight_delta = (model.HIST_model.rnn.weight_ih_l0 - initial["rnn.weight_ih_l0"]).abs().max().item()
    assert weight_delta > 0

    predictions = recorder.load_object("pred.pkl")
    labels = recorder.load_object("label.pkl")
    expected_index = pd.MultiIndex.from_product(
        [pd.to_datetime(context["test_dates"]), context["instruments"]], names=["datetime", "instrument"]
    )
    pd.testing.assert_index_equal(predictions.index, expected_index)
    pd.testing.assert_index_equal(labels.index, expected_index)
    assert predictions.shape == labels.shape == (48, 1)
    assert list(predictions.columns) == ["score"]
    assert np.isfinite(predictions.to_numpy()).all()
    assert np.isfinite(labels.to_numpy()).all()
    assert predictions["score"].groupby(level="datetime").std().gt(0).all()
    actual = model.predict(dataset).to_frame("score")
    pd.testing.assert_frame_equal(actual, predictions, check_exact=True)
    pd.testing.assert_frame_equal(SignalRecord.generate_label(dataset), labels, check_exact=True)

    for name in ("ic.pkl", "ric.pkl"):
        signal_analysis = recorder.load_object(f"sig_analysis/{name}")
        assert signal_analysis.shape == (6,)
        np.testing.assert_array_equal(signal_analysis.index, pd.to_datetime(context["test_dates"]))
        assert np.isfinite(signal_analysis.to_numpy()).all()
    metrics = recorder.list_metrics()
    assert np.isfinite([metrics[key] for key in ("IC", "ICIR", "Rank IC", "Rank ICIR")]).all()
    report = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    assert report.shape == (6, 9)
    pd.testing.assert_index_equal(report.index, pd.DatetimeIndex(context["backtest_dates"], name="datetime"))
    assert np.isfinite(report.to_numpy()).all()
    assert report["account"].gt(0).all() and report["value"].gt(0).all()
    assert report["turnover"].sum() > 0 and report["cost"].sum() > 0
    risk = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
    assert risk.shape == (10, 1)
    assert np.isfinite(risk.to_numpy()).all()

    # Opting in once must not change subsequent default loading.
    with pytest.raises(LoadObjectError, match="HIST"):
        recorder.load_object("params.pkl")
    with pytest.raises(LoadObjectError, match="DatasetH"):
        recorder.load_object("dataset")
    return {
        "prediction_rows": len(predictions),
        "backtest_rows": len(report),
        "optimizer_steps": 12,
        "weight_delta": weight_delta,
        "prediction_max_error": float((actual - predictions).abs().to_numpy().max()),
        "turnover": float(report["turnover"].sum()),
    }


@pytest.mark.slow
def test_hist_full_workflow_artifact_trust(workflow_context):
    context = workflow_context
    dates = context.calendar.strftime("%Y-%m-%d").tolist()
    inputs = {
        "root": str(context.root),
        "provider_uri": str(context.provider_uri),
        "uri": context.uri,
        "instruments": context.instruments,
        "segments": {
            "train": [dates[100], dates[111]],
            "valid": [dates[112], dates[115]],
            "test": [dates[116], dates[121]],
        },
        "test_dates": dates[116:122],
        "backtest_dates": dates[117:123],
    }
    source = Path(__file__).resolve()
    environment = dict(os.environ)
    environment.update(
        {
            "HOME": str(context.root),
            "USERPROFILE": str(context.root),
            "PYTHONUSERBASE": site.getuserbase(),
            "PYTHONPATH": os.pathsep.join(filter(None, [str(source.parents[2]), environment.get("PYTHONPATH")])),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MLFLOW_ALLOW_FILE_STORE": "true",
        }
    )
    # Isolate Torch RNG/threads and HIST's default ~/tmp checkpoint without replacing any workflow stages.
    results = []
    for stage in ("train", "reload"):
        result = subprocess.run(
            [sys.executable, str(source), stage, json.dumps(inputs)],
            cwd=context.root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert result.returncode == 0, f"{stage} failed:\n{result.stdout}\n{result.stderr}"
        results.append(json.loads((context.root / f"{stage}_result.json").read_text(encoding="utf-8")))
    assert results[0] == results[1]
    assert results[1]["prediction_rows"] == 48
    assert results[1]["backtest_rows"] == 6
    assert results[1]["optimizer_steps"] == 12
    assert results[1]["prediction_max_error"] == 0


if __name__ == "__main__":
    from qlib.workflow import R

    stage, encoded_context = sys.argv[1:]
    context = json.loads(encoded_context)
    root = Path(context["root"])
    _init_workflow(context)
    if stage == "train":
        identifiers = _train_workflow(context)
        (root / "recorder.json").write_text(json.dumps(identifiers), encoding="utf-8")
    else:
        assert stage == "reload"
        assert "qlib.contrib.model.pytorch_hist" not in sys.modules
        identifiers = json.loads((root / "recorder.json").read_text(encoding="utf-8"))
    recorder = R.get_recorder(recorder_id=identifiers[0], experiment_id=identifiers[1])
    result = _check_artifacts(context, recorder)
    assert "qlib.contrib.model.pytorch_hist" in sys.modules
    (root / f"{stage}_result.json").write_text(json.dumps(result), encoding="utf-8")

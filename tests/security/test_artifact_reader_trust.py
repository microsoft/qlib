import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from qlib.contrib.model.linear import LinearModel
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import StaticDataLoader
from qlib.utils.exceptions import LoadObjectError
from qlib.workflow import R
from qlib.workflow.record_temp import RecordTemp
from qlib.workflow.recorder import Recorder, UnsafeArtifactWarning
from qlib.workflow.task.collect import RecorderCollector


class NestedRecord(RecordTemp):
    artifact_path = "nested"
    depend_cls = RecordTemp


@pytest.fixture
def reader_artifacts(workflow_context):
    dates = workflow_context.calendar[:8]
    index = pd.MultiIndex.from_product([dates, ["SH600000"]], names=["datetime", "instrument"])
    x = np.arange(8, dtype=float)
    data = pd.DataFrame(
        np.column_stack([x, x**2, 2 * x - 0.3 * x**2 + 1]),
        index=index,
        columns=pd.MultiIndex.from_tuples([("feature", "x"), ("feature", "x2"), ("label", "y")]),
    )
    handler = DataHandlerLP(instruments=None, data_loader=StaticDataLoader(data))
    handler.config(dump_all=True)
    dataset = DatasetH(handler=handler, segments={"train": (dates[0], dates[-1]), "test": (dates[0], dates[-1])})
    model = LinearModel(estimator="ridge", alpha=0.1, fit_intercept=True)
    model.fit(dataset)
    prediction = model.predict(dataset)
    assert prediction.nunique() > 1
    with R.start(experiment_name="artifact-readers"):
        recorder = R.get_recorder()
        recorder.save_objects(
            **{
                "params.pkl": model,
                "dataset": dataset,
                "pred.pkl": prediction.to_frame("score"),
                "not-data.pkl": model,
            }
        )
    return SimpleNamespace(recorder=recorder, dataset=dataset, prediction=prediction)


@pytest.mark.parametrize("nested", [False, True])
def test_record_template_authorizes_only_the_requested_load(reader_artifacts, nested):
    artifacts = reader_artifacts
    record = NestedRecord(artifacts.recorder) if nested else RecordTemp(artifacts.recorder)
    with pytest.raises(LoadObjectError, match="LinearModel"):
        record.load("params.pkl")
    assert type(record) is (NestedRecord if nested else RecordTemp)
    with pytest.warns(UnsafeArtifactWarning):
        model = record.load("params.pkl", trusted=True)
    with pytest.warns(UnsafeArtifactWarning):
        dataset = record.load("dataset", trusted=True)
    pd.testing.assert_series_equal(model.predict(dataset), artifacts.prediction, check_exact=True)
    pd.testing.assert_frame_equal(record.load("pred.pkl"), artifacts.prediction.to_frame("score"), check_exact=True)
    with pytest.raises(LoadObjectError, match="LinearModel"):
        record.load("not-data.pkl")
    assert type(record) is (NestedRecord if nested else RecordTemp)


def test_record_template_can_disable_parent_lookup(reader_artifacts):
    record = NestedRecord(reader_artifacts.recorder)
    with pytest.raises(LoadObjectError):
        record.load("params.pkl", parents=False, trusted=True)
    assert type(record) is NestedRecord


def test_record_template_keeps_its_class_when_parent_artifact_is_missing(reader_artifacts):
    record = NestedRecord(reader_artifacts.recorder)
    with pytest.raises(LoadObjectError):
        record.load("missing.pkl", trusted=True)
    assert type(record) is NestedRecord


@pytest.mark.parametrize("value", ["false", 0, 1, None, np.bool_(True)])
def test_record_template_rejects_invalid_consent_before_reading(value):
    recorder = Mock()
    with pytest.raises(TypeError, match="must be a bool"):
        RecordTemp(recorder).load("unused.pkl", trusted=value)
    recorder.load_object.assert_not_called()


def test_collector_authorizes_selected_artifacts_without_widening_data_reads(reader_artifacts):
    artifacts = reader_artifacts
    paths = {"model": "params.pkl", "pred": "pred.pkl"}
    options = {"model": {"trusted": True}}
    collector = RecorderCollector(lambda: [artifacts.recorder], artifacts_path=paths, artifact_load_kwargs=options)
    assert options == {"model": {"trusted": True}}
    options["model"]["trusted"] = False
    options["pred"] = {"trusted": True}

    with pytest.warns(UnsafeArtifactWarning) as caught:
        collected = collector.collect(only_exist=False)
    assert len(caught) == 1
    recorder_id = artifacts.recorder.info["id"]
    pd.testing.assert_series_equal(
        collected["model"][recorder_id].predict(artifacts.dataset), artifacts.prediction, check_exact=True
    )
    pd.testing.assert_frame_equal(
        collected["pred"][recorder_id], artifacts.prediction.to_frame("score"), check_exact=True
    )

    collector.artifacts_path["pred"] = "not-data.pkl"
    with pytest.warns(UnsafeArtifactWarning), pytest.raises(LoadObjectError, match="LinearModel"):
        collector.collect(only_exist=False)


def test_collector_default_refusal_reports_why_an_artifact_is_skipped(reader_artifacts, monkeypatch):
    from qlib.workflow.task import collect

    logger = Mock()
    monkeypatch.setattr(collect, "get_module_logger", lambda *args: logger)
    collector = RecorderCollector(lambda: [reader_artifacts.recorder], artifacts_path={"model": "params.pkl"})
    with pytest.raises(LoadObjectError, match="LinearModel"):
        collector.collect(only_exist=False)
    assert collector.collect() == {}
    assert "Forbidden class:" in logger.warning.call_args.args[0]
    assert "LinearModel" in logger.warning.call_args.args[0]


def test_collector_preserves_backend_unpickler_policy(reader_artifacts):
    artifacts = reader_artifacts
    collector = RecorderCollector(
        lambda: [artifacts.recorder],
        artifacts_path={"model": "params.pkl", "pred": "pred.pkl"},
        artifact_load_kwargs={"model": {"unpickler": pickle.Unpickler}},
    )
    with pytest.warns(UnsafeArtifactWarning) as caught:
        collected = collector.collect(only_exist=False)
    assert len(caught) == 1
    recorder_id = artifacts.recorder.info["id"]
    pd.testing.assert_series_equal(
        collected["model"][recorder_id].predict(artifacts.dataset), artifacts.prediction, check_exact=True
    )
    pd.testing.assert_frame_equal(
        collected["pred"][recorder_id], artifacts.prediction.to_frame("score"), check_exact=True
    )

    collector.artifact_load_kwargs["model"]["trusted"] = True
    with pytest.raises(ValueError, match="unpickler"):
        collector.collect(only_exist=False)


@pytest.mark.parametrize("value", ["false", 0, 1, None, np.bool_(True)])
def test_collector_rejects_invalid_consent_before_opening_an_experiment(monkeypatch, value):
    get_exp = Mock()
    monkeypatch.setattr(R, "get_exp", get_exp)
    with pytest.raises(TypeError, match="must be a bool"):
        RecorderCollector("unused", artifact_load_kwargs={"pred": {"trusted": value}})
    get_exp.assert_not_called()


@pytest.mark.parametrize(
    "options,error",
    [
        ([], TypeError),
        (False, TypeError),
        ({"pred": None}, TypeError),
        ({"pred": []}, TypeError),
        ({"unknown": {"trusted": True}}, ValueError),
        ({"__raw": {"trusted": True}}, ValueError),
    ],
)
def test_collector_rejects_invalid_loading_options(options, error):
    with pytest.raises(error):
        RecorderCollector(lambda: [], artifact_load_kwargs=options)


@pytest.mark.parametrize("options", [None, {"pred": {"trusted": False}}])
def test_artifact_readers_preserve_legacy_default_recorder_signature(options):
    expected = pd.DataFrame({"score": [0.5]})

    class LegacyRecorder:
        info = {"id": "legacy-reader"}
        status = Recorder.STATUS_FI

        def load_object(self, name):
            assert name == "pred.pkl"
            return expected

    recorder = LegacyRecorder()
    assert RecordTemp(recorder).load("pred.pkl", trusted=False) is expected
    collector = RecorderCollector(lambda: [recorder], artifact_load_kwargs=options)
    assert collector.collect(only_exist=False)["pred"]["legacy-reader"] is expected


def test_collector_raw_records_do_not_require_deserialization(reader_artifacts):
    collector = RecorderCollector(lambda: [reader_artifacts.recorder], artifacts_key="__raw")
    assert collector.collect()["__raw"][reader_artifacts.recorder.info["id"]] is reader_artifacts.recorder

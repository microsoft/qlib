import pickle
import warnings
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import numpy as np
import pytest

from qlib.utils.exceptions import LoadObjectError
from qlib.utils.pickle_utils import ARTIFACT_MIGRATION_URL
from qlib.workflow.recorder import MLflowRecorder, Recorder, UnsafeArtifactWarning


class _TrackingClient:
    @staticmethod
    def _get_artifact_repo(_run_id):
        return object()


class _ArtifactClient:
    _tracking_client = _TrackingClient()

    def __init__(self, path):
        self.path = path

    def download_artifacts(self, _run_id, _name):
        return str(self.path)


class _CustomArtifact:
    def __init__(self, value=42):
        self.value = value


def _recorder(path):
    recorder = object.__new__(MLflowRecorder)
    recorder._uri = "file:///unused"
    recorder.id = "run-id"
    recorder.client = _ArtifactClient(path)
    return recorder


@pytest.mark.parametrize("options", [{}, {"trusted": False}])
def test_mlflow_artifact_uses_restricted_loading_without_fallback(tmp_path, monkeypatch, options):
    from qlib.workflow import recorder as recorder_module

    path = tmp_path / "payload.pkl"
    path.write_bytes(pickle.dumps(_CustomArtifact()))
    unrestricted = Mock(side_effect=AssertionError("Unrestricted loading must not be attempted"))
    monkeypatch.setattr(recorder_module.pickle, "Unpickler", unrestricted)

    with warnings.catch_warnings(record=True) as caught:
        with pytest.raises(LoadObjectError, match="Forbidden class") as error:
            _recorder(path).load_object("payload.pkl", **options)

    assert "payload.pkl" in str(error.value)
    assert "trusted=True" in str(error.value)
    assert "workflow entry point" in str(error.value)
    assert ARTIFACT_MIGRATION_URL in str(error.value)
    unrestricted.assert_not_called()
    assert not any(issubclass(warning.category, UnsafeArtifactWarning) for warning in caught)


@pytest.mark.parametrize("protocol", [4, 5])
def test_mlflow_artifact_loads_common_data_without_trusted_flag(tmp_path, protocol):
    path = tmp_path / "frame.pkl"
    expected = pd.DataFrame({"value": [1, 2]}, index=pd.date_range("2024-01-01", periods=2))
    path.write_bytes(pickle.dumps(expected, protocol=protocol))

    actual = _recorder(path).load_object("frame.pkl")

    pd.testing.assert_frame_equal(actual, expected)


def test_mlflow_artifact_requires_explicit_trust_for_arbitrary_pickle(tmp_path):
    path = tmp_path / "payload.pkl"
    path.write_bytes(pickle.dumps(_CustomArtifact()))
    recorder = _recorder(path)

    with pytest.warns(UnsafeArtifactWarning, match="may execute arbitrary code"):
        actual = recorder.load_object("payload.pkl", trusted=True)
    assert isinstance(actual, _CustomArtifact)
    assert actual.value == 42
    with pytest.raises(LoadObjectError, match="Forbidden class"):
        recorder.load_object("payload.pkl")


def test_custom_artifact_unpickler_requires_explicit_selection_and_warns(tmp_path):
    path = tmp_path / "payload.pkl"
    path.write_bytes(pickle.dumps(_CustomArtifact()))
    with pytest.warns(UnsafeArtifactWarning, match="custom artifact unpickler"):
        actual = _recorder(path).load_object("payload.pkl", unpickler=pickle.Unpickler)
    assert isinstance(actual, _CustomArtifact)
    assert actual.value == 42


def test_artifact_loader_rejects_ambiguous_trust_options(tmp_path):
    with pytest.raises(ValueError, match="cannot be used together"):
        _recorder(tmp_path / "unused.pkl").load_object("unused.pkl", unpickler=pickle.Unpickler, trusted=True)


@pytest.mark.parametrize("trusted", [None, 0, 1, "False", "True", np.bool_(True)])
def test_mlflow_artifact_requires_an_actual_boolean_before_download(tmp_path, trusted):
    recorder = _recorder(tmp_path / "unused.pkl")
    recorder.client.download_artifacts = Mock()

    with pytest.raises(TypeError, match="bool"):
        recorder.load_object("unused.pkl", trusted=trusted)

    recorder.client.download_artifacts.assert_not_called()


def _facade(recorder):
    from qlib.workflow import QlibRecorder

    experiment = SimpleNamespace(get_recorder=Mock(return_value=recorder))
    facade = object.__new__(QlibRecorder)
    facade.get_exp = Mock(return_value=experiment)
    return facade


@pytest.mark.parametrize("options", [{}, {"trusted": False}])
def test_recorder_facade_preserves_legacy_subclass_signature(options):
    class LegacyRecorder(Recorder):
        def load_object(self, name):
            return {"name": name}

    recorder = Mock(wraps=LegacyRecorder("experiment", "legacy"))

    assert _facade(recorder).load_object("data.pkl", **options) == {"name": "data.pkl"}
    recorder.load_object.assert_called_once_with("data.pkl")


def test_recorder_facade_forwards_explicit_trust_to_modern_subclass():
    class ModernRecorder(Recorder):
        def load_object(self, name, *, trusted=False):
            return {"name": name, "trusted": trusted}

    recorder = Mock(wraps=ModernRecorder("experiment", "modern"))

    assert _facade(recorder).load_object("model.pkl", trusted=True) == {"name": "model.pkl", "trusted": True}
    recorder.load_object.assert_called_once_with("model.pkl", trusted=True)


@pytest.fixture
def mlflow_recorders(tmp_path):
    from mlflow.tracking import MlflowClient

    uri = (tmp_path / "mlruns").as_uri()
    client = MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment("artifact-trust")
    run = client.create_run(experiment_id, tags={"mlflow.runName": "roundtrip"})
    writer = MLflowRecorder(experiment_id, uri, mlflow_run=run)
    reader = MLflowRecorder(experiment_id, uri, mlflow_run=client.get_run(run.info.run_id))
    try:
        yield writer, reader
    finally:
        client.set_terminated(run.info.run_id)


@pytest.mark.parametrize(
    "value",
    [
        pd.Series([1, 2], index=pd.period_range("2024-01", periods=2, freq="M")),
        pd.Series([1, 2], index=pd.IntervalIndex.from_breaks([0, 1, 2])),
        pd.Series([0.0, 1.0, 0.0], dtype=pd.SparseDtype("float64", 0)),
        np.ma.array([1, 2, 3], mask=[False, True, False]),
    ],
)
def test_real_mlflow_store_roundtrips_data_artifacts(mlflow_recorders, value):
    writer, reader = mlflow_recorders
    writer.save_objects(**{"data.pkl": value})
    # A new recorder/client must download and deserialize the stored artifact.
    actual = reader.load_object("data.pkl")
    if isinstance(value, pd.Series):
        pd.testing.assert_series_equal(actual, value)
    else:
        np.testing.assert_array_equal(actual.data, value.data)
        np.testing.assert_array_equal(actual.mask, value.mask)


def test_real_mlflow_model_and_dataset_require_workflow_opt_in(mlflow_recorders):
    from qlib.contrib.model.linear import LinearModel
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import DataLoaderDH
    from qlib.workflow.online.update import RMDLoader

    dates = pd.date_range("2024-01-01", periods=6)
    index = pd.MultiIndex.from_product([dates, ["SH600000", "SH600004"]], names=["datetime", "instrument"])
    feature = np.arange(len(index), dtype=float)
    frame = pd.DataFrame(
        {("feature", "value"): feature, ("label", "LABEL0"): 2 * feature + 1},
        index=index,
    )
    source = DataHandlerLP.from_df(frame)
    source.config(dump_all=True)
    handler = DataHandlerLP(data_loader=DataLoaderDH(source))
    dataset = DatasetH(handler, segments={"train": (dates[0], dates[3]), "test": (dates[4], dates[5])})
    model = LinearModel(fit_intercept=True).fit(dataset)
    expected = model.predict(dataset)
    writer, reader = mlflow_recorders
    writer.save_objects(**{"params.pkl": model, "dataset": dataset, "pred.pkl": expected.to_frame("score")})
    reader.load_object = Mock(wraps=reader.load_object)

    legacy_loader = object.__new__(RMDLoader)
    legacy_loader.rec = reader
    for default_loader in (RMDLoader(reader), legacy_loader):
        assert default_loader.trusted is False
        with pytest.raises(LoadObjectError, match="LinearModel"):
            default_loader.get_model()
        with pytest.raises(LoadObjectError, match="DatasetH"):
            default_loader.get_dataset(dates[4], dates[5])

    trusted_loader = RMDLoader(reader, trusted=True)
    with pytest.warns(UnsafeArtifactWarning):
        loaded_model = trusted_loader.get_model()
        loaded_dataset = trusted_loader.get_dataset(dates[4], dates[5])

    assert isinstance(loaded_model, LinearModel)
    assert isinstance(loaded_dataset, DatasetH)
    assert loaded_dataset.segments == {"test": (dates[4], dates[5])}
    pd.testing.assert_series_equal(loaded_model.predict(loaded_dataset), expected)
    np.testing.assert_allclose(expected.values, frame.loc[dates[4] :, ("label", "LABEL0")].values)
    pd.testing.assert_frame_equal(reader.load_object("pred.pkl"), expected.to_frame("score"))
    reader.load_object.assert_called_with("pred.pkl")


@pytest.mark.parametrize("updater_name,artifact_name", [("PredUpdater", "pred.pkl"), ("LabelUpdater", "label.pkl")])
def test_trusted_workflow_does_not_trust_prediction_or_label_artifacts(
    mlflow_recorders, monkeypatch, updater_name, artifact_name
):
    from qlib.workflow.online import update

    writer, reader = mlflow_recorders
    writer.save_objects(**{artifact_name: _CustomArtifact()})
    reader.load_object = Mock(wraps=reader.load_object)
    monkeypatch.setattr(update, "D", SimpleNamespace(calendar=lambda **kwargs: pd.date_range("2024-01-01", periods=2)))

    with pytest.raises(LoadObjectError, match="Forbidden class"):
        getattr(update, updater_name)(reader, trusted=True)

    reader.load_object.assert_called_once_with(artifact_name)


def test_end_task_train_keeps_data_only_tasks_usable_by_default(tmp_path, monkeypatch):
    from qlib.model import trainer

    task = {"model": {"class": "LinearModel", "module_path": "qlib.contrib.model.linear"}}
    path = tmp_path / "task.pkl"
    path.write_bytes(pickle.dumps(task))
    recorder = _recorder(path)
    execute = Mock()
    monkeypatch.setattr(
        trainer,
        "R",
        SimpleNamespace(start=Mock(return_value=nullcontext()), load_object=recorder.load_object),
    )
    monkeypatch.setattr(trainer, "_exe_task", execute)
    record_info = SimpleNamespace(info={"id": recorder.id})

    with warnings.catch_warnings(record=True) as caught:
        assert trainer.end_task_train(record_info, "training") is record_info

    execute.assert_called_once_with(task)
    assert not any(issubclass(warning.category, UnsafeArtifactWarning) for warning in caught)


def test_end_task_train_loads_trusted_reweighter(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from qlib.contrib.meta.data_selection.model import TimeReweighter
    from qlib.model import trainer

    weights = pd.Series([0.5, 1.0])
    path = tmp_path / "task.pkl"
    path.write_bytes(pickle.dumps({"reweighter": TimeReweighter(weights)}))
    recorder = _recorder(path)
    with pytest.raises(LoadObjectError, match="TimeReweighter"):
        recorder.load_object("task")

    start = Mock(return_value=nullcontext())
    load = Mock(wraps=recorder.load_object)
    execute = Mock()
    monkeypatch.setattr(trainer, "R", SimpleNamespace(start=start, load_object=load))
    monkeypatch.setattr(trainer, "_exe_task", execute)
    record_info = SimpleNamespace(info={"id": recorder.id})

    with pytest.raises(LoadObjectError, match="trusted=True"):
        trainer.end_task_train(record_info, "training")
    execute.assert_not_called()
    start.reset_mock()
    load.reset_mock()

    with pytest.warns(UnsafeArtifactWarning):
        assert trainer.end_task_train(record_info, "training", trusted=True) is record_info

    start.assert_called_once_with(experiment_name="training", recorder_id=recorder.id, resume=True)
    load.assert_called_once_with("task", trusted=True)
    execute.assert_called_once()
    loaded = execute.call_args[0][0]["reweighter"]
    assert isinstance(loaded, TimeReweighter)
    pd.testing.assert_series_equal(loaded.time_weight, weights)


@pytest.mark.parametrize("options", [{}, {"trusted": False}, {"trusted": True}])
def test_ddgda_requires_opt_in_before_meta_model_inference(tmp_path, monkeypatch, options):
    pytest.importorskip("torch")
    from qlib.contrib.rolling import ddgda

    model = ddgda.MetaModelDS(step=20, hist_step_n=3)
    model.fitted = True
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps(model))
    recorder = _recorder(path)
    with pytest.raises(LoadObjectError, match="MetaModelDS"):
        recorder.load_object("model")

    recorder.list_params = Mock(return_value={"trunc_days": "1", "step": "20", "hist_step_n": "3"})
    recorder.load_object = Mock(wraps=recorder.load_object)
    experiment = SimpleNamespace(RT_L="list", list_recorders=Mock(return_value=[recorder]))
    monkeypatch.setattr(ddgda, "R", SimpleNamespace(get_exp=Mock(return_value=experiment)))
    monkeypatch.setattr(ddgda.Rolling, "__init__", Mock(return_value=None))
    monkeypatch.setattr(ddgda.Rolling, "get_task_list", Mock(return_value=[]))
    meta_dataset = object()
    monkeypatch.setattr(ddgda, "MetaDatasetDS", Mock(return_value=meta_dataset))
    inference_calls = Mock()

    def inference(loaded_model, dataset):
        inference_calls(loaded_model, dataset)
        assert loaded_model.fitted
        assert loaded_model.step == 20
        assert dataset is meta_dataset
        return [{"generated": True}]

    monkeypatch.setattr(ddgda.MetaModelDS, "inference", inference)
    rolling = ddgda.DDGDA(working_dir=tmp_path, **options)
    rolling.step = 20
    rolling._internal_data_path.write_bytes(pickle.dumps(None))

    if not options.get("trusted", False):
        with pytest.raises(LoadObjectError, match="trusted=True"):
            rolling.get_task_list()
        inference_calls.assert_not_called()
        ddgda.MetaDatasetDS.assert_not_called()
        assert not rolling._task_path.exists()
        return

    with pytest.warns(UnsafeArtifactWarning):
        assert rolling.get_task_list() == [{"generated": True}]

    recorder.load_object.assert_called_once_with("model", trusted=True)
    assert pickle.loads(rolling._task_path.read_bytes()) == [{"generated": True}]


@pytest.mark.parametrize("options", [{}, {"trusted": False}, {"trusted": True}])
def test_rolling_strategy_requires_opt_in_for_tasks_with_reweighters(tmp_path, monkeypatch, options):
    pytest.importorskip("torch")
    from qlib.contrib.meta.data_selection.model import TimeReweighter
    from qlib.workflow.online import strategy

    segment = (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31"))
    task = {
        "dataset": {"kwargs": {"segments": {"test": segment}}},
        "reweighter": TimeReweighter(pd.Series([1.0])),
    }
    path = tmp_path / "task.pkl"
    path.write_bytes(pickle.dumps(task))
    recorder = _recorder(path)
    monkeypatch.setattr(strategy, "TimeAdjuster", Mock())
    rolling = strategy.RollingStrategy("rolling", task, object.__new__(strategy.RollingGen), **options)

    if not options.get("trusted", False):
        with pytest.raises(LoadObjectError, match="trusted=True"):
            rolling._list_latest([recorder])
        return

    with pytest.warns(UnsafeArtifactWarning):
        records, latest = rolling._list_latest([recorder])
    assert records == [recorder]
    assert latest == segment

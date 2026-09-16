import pickle
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from qlib.utils.exceptions import LoadObjectError
from qlib.workflow.recorder import MLflowRecorder, UnsafeArtifactWarning


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


class _MaliciousPayload:
    def __reduce__(self):
        return eval, ("40 + 2",)


def _recorder(path):
    recorder = object.__new__(MLflowRecorder)
    recorder._uri = "file:///unused"
    recorder.id = "run-id"
    recorder.client = _ArtifactClient(path)
    return recorder


def test_mlflow_artifact_uses_restricted_loading_by_default(tmp_path):
    path = tmp_path / "payload.pkl"
    path.write_bytes(pickle.dumps(_MaliciousPayload()))

    with pytest.raises(LoadObjectError, match="Forbidden class"):
        _recorder(path).load_object("payload.pkl")


@pytest.mark.parametrize("protocol", [4, 5])
def test_mlflow_artifact_loads_common_data_without_trusted_flag(tmp_path, protocol):
    path = tmp_path / "frame.pkl"
    expected = pd.DataFrame({"value": [1, 2]}, index=pd.date_range("2024-01-01", periods=2))
    path.write_bytes(pickle.dumps(expected, protocol=protocol))

    actual = _recorder(path).load_object("frame.pkl")

    pd.testing.assert_frame_equal(actual, expected)


def test_mlflow_artifact_requires_explicit_trust_for_arbitrary_pickle(tmp_path):
    path = tmp_path / "payload.pkl"
    path.write_bytes(pickle.dumps(_MaliciousPayload()))

    with pytest.warns(UnsafeArtifactWarning, match="may execute arbitrary code"):
        assert _recorder(path).load_object("payload.pkl", trusted=True) == 42


def test_custom_artifact_unpickler_requires_explicit_selection_and_warns(tmp_path):
    path = tmp_path / "payload.pkl"
    path.write_bytes(pickle.dumps(_MaliciousPayload()))
    with pytest.warns(UnsafeArtifactWarning, match="custom artifact unpickler"):
        assert _recorder(path).load_object("payload.pkl", unpickler=pickle.Unpickler) == 42


def test_artifact_loader_rejects_ambiguous_trust_options(tmp_path):
    with pytest.raises(ValueError, match="cannot be used together"):
        _recorder(tmp_path / "unused.pkl").load_object("unused.pkl", unpickler=pickle.Unpickler, trusted=True)


@pytest.mark.parametrize("trusted", [False, True])
def test_recorder_facade_forwards_explicit_trust(trusted):
    from qlib.workflow import QlibRecorder

    recorder = Mock()
    experiment = SimpleNamespace(get_recorder=Mock(return_value=recorder))
    facade = object.__new__(QlibRecorder)
    facade.get_exp = Mock(return_value=experiment)
    facade.load_object("model.pkl", trusted=trusted)
    recorder.load_object.assert_called_once_with("model.pkl", trusted=trusted)


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

    with pytest.warns(UnsafeArtifactWarning):
        assert trainer.end_task_train(record_info, "training") is record_info

    start.assert_called_once_with(experiment_name="training", recorder_id=recorder.id, resume=True)
    load.assert_called_once_with("task", trusted=True)
    execute.assert_called_once()
    loaded = execute.call_args[0][0]["reweighter"]
    assert isinstance(loaded, TimeReweighter)
    pd.testing.assert_series_equal(loaded.time_weight, weights)


def test_ddgda_loads_trusted_meta_model_before_inference(tmp_path, monkeypatch):
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
    monkeypatch.setattr(ddgda.Rolling, "get_task_list", Mock(return_value=[]))
    meta_dataset = object()
    monkeypatch.setattr(ddgda, "MetaDatasetDS", Mock(return_value=meta_dataset))

    def inference(loaded_model, dataset):
        assert loaded_model.fitted
        assert loaded_model.step == 20
        assert dataset is meta_dataset
        return [{"generated": True}]

    monkeypatch.setattr(ddgda.MetaModelDS, "inference", inference)
    rolling = object.__new__(ddgda.DDGDA)
    rolling.meta_exp_name = "DDG-DA"
    rolling.working_dir = tmp_path
    rolling.step = 20
    rolling._internal_data_path.write_bytes(pickle.dumps(None))

    with pytest.warns(UnsafeArtifactWarning):
        assert rolling.get_task_list() == [{"generated": True}]

    recorder.load_object.assert_called_once_with("model", trusted=True)
    assert pickle.loads(rolling._task_path.read_bytes()) == [{"generated": True}]


def test_rolling_strategy_reads_tasks_with_reweighters(tmp_path):
    pytest.importorskip("torch")
    from qlib.contrib.meta.data_selection.model import TimeReweighter
    from qlib.workflow.online.strategy import RollingStrategy

    segment = (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31"))
    task = {
        "dataset": {"kwargs": {"segments": {"test": segment}}},
        "reweighter": TimeReweighter(pd.Series([1.0])),
    }
    path = tmp_path / "task.pkl"
    path.write_bytes(pickle.dumps(task))
    recorder = _recorder(path)
    strategy = object.__new__(RollingStrategy)

    with pytest.warns(UnsafeArtifactWarning):
        records, latest = strategy._list_latest([recorder])
    assert records == [recorder]
    assert latest == segment

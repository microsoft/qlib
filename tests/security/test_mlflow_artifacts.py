import pickle

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


def test_mlflow_artifact_loads_common_data_without_trusted_flag(tmp_path):
    path = tmp_path / "frame.pkl"
    expected = pd.DataFrame({"value": [1, 2]})
    path.write_bytes(pickle.dumps(expected))

    actual = _recorder(path).load_object("frame.pkl")

    pd.testing.assert_frame_equal(actual, expected)


def test_mlflow_artifact_requires_explicit_trust_for_arbitrary_pickle(tmp_path):
    path = tmp_path / "payload.pkl"
    path.write_bytes(pickle.dumps(_MaliciousPayload()))

    with pytest.warns(UnsafeArtifactWarning, match="may execute arbitrary code"):
        assert _recorder(path).load_object("payload.pkl", trusted=True) == 42

import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class WorkflowExpManagerLockPathTest(unittest.TestCase):
    @staticmethod
    def _load_expm_module():
        expm_path = Path(__file__).resolve().parents[1] / "qlib" / "workflow" / "expm.py"

        qlib_pkg = ModuleType("qlib")
        qlib_pkg.__path__ = []
        workflow_pkg = ModuleType("qlib.workflow")
        workflow_pkg.__path__ = []
        utils_pkg = ModuleType("qlib.utils")
        utils_pkg.__path__ = []

        mlflow_pkg = ModuleType("mlflow")
        exceptions_pkg = ModuleType("mlflow.exceptions")
        entities_pkg = ModuleType("mlflow.entities")
        filelock_pkg = ModuleType("filelock")
        exp_pkg = ModuleType("qlib.workflow.exp")
        config_pkg = ModuleType("qlib.config")
        recorder_pkg = ModuleType("qlib.workflow.recorder")
        log_pkg = ModuleType("qlib.log")
        utils_exceptions_pkg = ModuleType("qlib.utils.exceptions")

        mlflow_pkg.tracking = SimpleNamespace(MlflowClient=object)
        exceptions_pkg.MlflowException = type("MlflowException", (Exception,), {})
        exceptions_pkg.RESOURCE_ALREADY_EXISTS = "RESOURCE_ALREADY_EXISTS"
        exceptions_pkg.ErrorCode = type("ErrorCode", (), {"Name": staticmethod(lambda code: code)})
        entities_pkg.ViewType = type("ViewType", (), {"ACTIVE_ONLY": "ACTIVE_ONLY"})
        filelock_pkg.FileLock = type("FileLock", (), {})
        exp_pkg.MLflowExperiment = type("MLflowExperiment", (), {})
        exp_pkg.Experiment = type("Experiment", (), {})
        config_pkg.C = SimpleNamespace(exp_manager={})
        recorder_pkg.Recorder = type("Recorder", (), {"STATUS_S": "S"})
        log_pkg.get_module_logger = lambda name: SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None)
        utils_exceptions_pkg.ExpAlreadyExistError = type("ExpAlreadyExistError", (Exception,), {})

        with patch.dict(
            "sys.modules",
            {
                "qlib": qlib_pkg,
                "qlib.workflow": workflow_pkg,
                "qlib.utils": utils_pkg,
                "mlflow": mlflow_pkg,
                "mlflow.exceptions": exceptions_pkg,
                "mlflow.entities": entities_pkg,
                "filelock": filelock_pkg,
                "qlib.workflow.exp": exp_pkg,
                "qlib.config": config_pkg,
                "qlib.workflow.recorder": recorder_pkg,
                "qlib.log": log_pkg,
                "qlib.utils.exceptions": utils_exceptions_pkg,
            },
        ):
            spec = spec_from_file_location("qlib.workflow.expm", expm_path)
            module = module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module

    def test_get_or_create_exp_uses_absolute_filelock_path_for_file_uri(self):
        module = self._load_expm_module()
        captured = {}

        class DummyFileLock:
            def __init__(self, path):
                captured["path"] = Path(path)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class DummyExpManager(module.ExpManager):
            def _start_exp(self, *args, **kwargs):
                raise NotImplementedError

            def _end_exp(self, *args, **kwargs):
                raise NotImplementedError

            def create_exp(self, experiment_name=None):
                return experiment_name

            def _get_exp(self, experiment_id=None, experiment_name=None):
                raise ValueError("missing")

            def search_records(self, experiment_ids=None, **kwargs):
                raise NotImplementedError

        manager = DummyExpManager(uri="file:///tmp/qlib/mlruns", default_exp_name="Default")

        with patch.object(module, "FileLock", DummyFileLock):
            exp, created = manager._get_or_create_exp(experiment_name="exp")

        self.assertTrue(created)
        self.assertEqual("exp", exp)
        self.assertEqual(Path("/tmp/qlib/mlruns/filelock"), captured["path"])
        self.assertTrue(captured["path"].is_absolute())


if __name__ == "__main__":
    unittest.main()

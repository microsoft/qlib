import inspect
import pickle
import runpy
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
from fire.helptext import HelpText

from qlib.utils.pickle_utils import ARTIFACT_MIGRATION_URL, ArtifactTrustMixin


WORKFLOWS = [
    ("qlib.workflow.online.strategy", "RollingStrategy"),
    ("qlib.workflow.online.utils", "OnlineToolR"),
    ("qlib.workflow.online.update", "RMDLoader"),
    ("qlib.workflow.online.update", "PredUpdater"),
    ("qlib.workflow.online.update", "LabelUpdater"),
    ("qlib.model.trainer", "DelayTrainerR"),
    ("qlib.model.trainer", "DelayTrainerRM"),
    ("qlib.contrib.rolling.ddgda", "DDGDA"),
]


@pytest.fixture(params=WORKFLOWS, ids=[name for _, name in WORKFLOWS])
def workflow_cls(request):
    module, name = request.param
    return getattr(import_module(module), name)


@pytest.mark.parametrize("consent", [False, True])
def test_saved_pre_release_consent_migrates_once(workflow_cls, consent):
    original = object.__new__(workflow_cls)
    original.__dict__.update(trusted_artifacts=consent, marker="preserved")

    with pytest.warns(FutureWarning, match="Migrated pre-release") as caught:
        restored = pickle.loads(pickle.dumps(original))

    assert len(caught) == 1
    assert ARTIFACT_MIGRATION_URL in str(caught[0].message)
    assert restored.trusted is consent
    assert restored.__dict__ == {"trusted": consent, "marker": "preserved"}
    with warnings.catch_warnings(record=True) as caught_again:
        warnings.simplefilter("always")
        reloaded = pickle.loads(pickle.dumps(restored))
    assert not caught_again
    assert reloaded.__dict__ == restored.__dict__


def test_saved_workflow_without_consent_does_not_gain_it(workflow_cls):
    original = object.__new__(workflow_cls)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = pickle.loads(pickle.dumps(original))
    assert not caught
    assert restored.trusted is False
    assert "trusted" not in restored.__dict__


@pytest.mark.parametrize(
    "state,error",
    [
        ({"trusted": False, "trusted_artifacts": True}, ValueError),
        ({"trusted": True, "trusted_artifacts": False}, ValueError),
        ({"trusted_artifacts": "false"}, TypeError),
        ({"trusted_artifacts": 1}, TypeError),
        ({"trusted_artifacts": np.bool_(True)}, TypeError),
        ({"trusted": None}, TypeError),
    ],
)
def test_invalid_saved_consent_is_rejected_without_changing_state(workflow_cls, state, error):
    workflow = object.__new__(workflow_cls)
    workflow.marker = "preserved"
    with pytest.raises(error) as caught:
        workflow.__setstate__(state)
    assert ARTIFACT_MIGRATION_URL in str(caught.value)
    assert workflow.__dict__ == {"marker": "preserved"}
    assert workflow.trusted is False


class _StatefulBase:
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.base_restored = True


class _StatefulWorkflow(ArtifactTrustMixin, _StatefulBase):
    pass


def test_trust_migration_preserves_base_restore_hook():
    original = _StatefulWorkflow()
    original.trusted_artifacts = True
    with pytest.warns(FutureWarning):
        restored = pickle.loads(pickle.dumps(original))
    assert restored.base_restored
    assert restored.trusted is True
    assert "trusted_artifacts" not in restored.__dict__


def test_nested_online_manager_preserves_each_components_consent():
    from qlib.model.trainer import DelayTrainerR
    from qlib.workflow.online.manager import OnlineManager
    from qlib.workflow.online.strategy import RollingStrategy
    from qlib.workflow.online.utils import OnlineToolR

    manager = object.__new__(OnlineManager)
    strategy = object.__new__(RollingStrategy)
    strategy.trusted_artifacts = True
    strategy.tool = object.__new__(OnlineToolR)
    strategy.tool.trusted_artifacts = False
    manager.strategies = [strategy]
    manager.trainer = object.__new__(DelayTrainerR)
    manager.trainer.trusted_artifacts = True

    with pytest.warns(FutureWarning) as caught:
        restored = pickle.loads(pickle.dumps(manager))

    assert len(caught) == 3
    assert restored.strategies[0].trusted is True
    assert restored.strategies[0].tool.trusted is False
    assert restored.trainer.trusted is True


def test_public_workflow_constructors_expose_only_trusted(workflow_cls):
    parameters = inspect.signature(workflow_cls).parameters
    assert parameters["trusted"].default is False
    assert "trusted_artifacts" not in parameters


def _entry_points():
    from qlib.contrib.meta.data_selection.dataset import InternalData, MetaDatasetDS
    from qlib.contrib.rolling.ddgda import DDGDA
    from qlib.model.trainer import DelayTrainerR, DelayTrainerRM, end_task_train
    from qlib.workflow import QlibRecorder
    from qlib.workflow.online.strategy import RollingStrategy
    from qlib.workflow.online.update import LabelUpdater, PredUpdater, RMDLoader
    from qlib.workflow.online.utils import OnlineToolR
    from qlib.workflow.recorder import MLflowRecorder, Recorder

    return [
        (RMDLoader, {"rec": None}),
        (PredUpdater, {"record": None}),
        (LabelUpdater, {"record": None}),
        (OnlineToolR, {}),
        (RollingStrategy, {"name_id": "unused", "task_template": {}, "rolling_gen": None}),
        (DelayTrainerR, {}),
        (DelayTrainerRM, {}),
        (DDGDA, {"conf_path": "unused.yaml"}),
        (MetaDatasetDS, {"task_tpl": [], "step": 20, "exp_name": "unused", "segments": 0.5}),
        (InternalData({}, 20, "unused").setup, {}),
        (end_task_train, {"rec": None, "experiment_name": "unused"}),
        (QlibRecorder(None).load_object, {"name": "unused"}),
        (MLflowRecorder.load_object, {}),
        (Recorder.load_object, {}),
    ]


def test_all_entry_points_use_the_same_public_keyword():
    for entry, _ in _entry_points():
        parameters = inspect.signature(entry).parameters
        assert parameters["trusted"].default is False, entry
        assert "trusted_artifacts" not in parameters, entry


@pytest.mark.parametrize("value", ["false", 0, 1, None, np.bool_(True)])
def test_workflow_entries_reject_non_boolean_consent_before_loading(value):
    for entry, kwargs in _entry_points()[:-2]:
        with pytest.raises(TypeError, match="`trusted` must be a bool"):
            entry(**kwargs, trusted=value)


def test_public_entries_do_not_accept_the_pre_release_alias():
    for entry, kwargs in _entry_points()[:-2]:
        with pytest.raises(TypeError, match="trusted_artifacts"):
            entry(**kwargs, trusted_artifacts=True)


@pytest.mark.parametrize(
    "path,class_name",
    [
        ("examples/online_srv/update_online_pred.py", "UpdatePredExample"),
        ("examples/online_srv/online_management_simulate.py", "OnlineSimulationExample"),
        ("examples/online_srv/rolling_online_management.py", "RollingOnlineExample"),
        ("examples/benchmarks_dynamic/DDG-DA/workflow.py", "DDGDABench"),
    ],
)
def test_example_cli_help_exposes_the_same_trust_flag(path, class_name):
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[2] / path))
    example = namespace[class_name]
    assert inspect.signature(example).parameters["trusted"].default is False
    text = HelpText(example)
    assert "--trusted" in text
    assert "--trusted_artifacts" not in text


@pytest.mark.parametrize("consent", [False, True])
def test_saved_ddgda_cache_config_migrates_without_changing_consent(tmp_path, consent):
    from qlib.contrib.meta.data_selection.dataset import InternalData
    from qlib.utils import init_instance_by_config
    from qlib.workflow.recorder import UnsafeArtifactWarning

    path = tmp_path / "internal.pkl"
    path.write_bytes(pickle.dumps(InternalData({}, 20, "unused")))
    config = {
        "class": "qlib.contrib.rolling.ddgda._load_cache",
        "kwargs": {"path": str(path), "trusted_artifacts": consent},
    }
    restored_config = pickle.loads(pickle.dumps(config))
    if consent:
        with pytest.warns(FutureWarning), pytest.warns(UnsafeArtifactWarning):
            result = init_instance_by_config(restored_config)
        assert isinstance(result, InternalData)
    else:
        with pytest.warns(FutureWarning), pytest.raises(pickle.UnpicklingError, match="trusted=True") as caught:
            init_instance_by_config(restored_config)
        assert ARTIFACT_MIGRATION_URL in str(caught.value)


@pytest.mark.parametrize(
    "options,error",
    [
        ({"trusted": False, "trusted_artifacts": True}, ValueError),
        ({"trusted": True, "trusted_artifacts": False}, ValueError),
        ({"trusted_artifacts": "false"}, TypeError),
        ({"trusted_artifacts": 0}, TypeError),
        ({"unknown_option": True}, TypeError),
    ],
)
def test_invalid_legacy_cache_options_fail_before_opening(tmp_path, options, error):
    from qlib.contrib.rolling.ddgda import _load_cache

    with pytest.raises(error):
        _load_cache(tmp_path / "not-opened.pkl", **options)


def test_regenerated_ddgda_task_drops_old_keyword_and_uses_selected_policy(tmp_path):
    from qlib.contrib.rolling.ddgda import DDGDA

    task = {
        "dataset": {
            "kwargs": {
                "handler": {
                    "class": "qlib.contrib.rolling.ddgda._load_cache",
                    "kwargs": {"path": str(tmp_path / "handler.pkl"), "trusted_artifacts": True},
                }
            }
        }
    }
    workflow = object.__new__(DDGDA)
    with pytest.warns(FutureWarning):
        result = workflow._replace_handler_with_cache(task)
    assert result["dataset"]["kwargs"]["handler"]["kwargs"] == {
        "path": str(tmp_path / "handler.pkl"),
        "trusted": False,
    }
    assert b"trusted_artifacts" not in pickle.dumps(result)

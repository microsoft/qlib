import inspect
import pickle
import runpy
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
from fire.helptext import HelpText

from qlib.utils.pickle_utils import ARTIFACT_MIGRATION_URL


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
def test_saved_consent_survives_restoring_a_workflow(workflow_cls, consent):
    original = object.__new__(workflow_cls)
    original.trusted = consent
    original.marker = "preserved"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = pickle.loads(pickle.dumps(original))
    assert not caught
    assert restored.trusted is consent
    assert restored.__dict__ == {"trusted": consent, "marker": "preserved"}


def test_saved_workflow_without_consent_does_not_gain_it(workflow_cls):
    original = object.__new__(workflow_cls)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = pickle.loads(pickle.dumps(original))
    assert not caught
    assert restored.trusted is False
    assert "trusted" not in restored.__dict__


def test_old_online_manager_requires_consent_on_each_component():
    from qlib.model.trainer import DelayTrainerR
    from qlib.workflow.online.manager import OnlineManager
    from qlib.workflow.online.strategy import RollingStrategy
    from qlib.workflow.online.utils import OnlineToolR

    manager = object.__new__(OnlineManager)
    strategy = object.__new__(RollingStrategy)
    strategy.tool = object.__new__(OnlineToolR)
    manager.strategies = [strategy]
    manager.trainer = object.__new__(DelayTrainerR)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = pickle.loads(pickle.dumps(manager))

    assert not caught
    assert restored.strategies[0].trusted is False
    assert restored.strategies[0].tool.trusted is False
    assert restored.trainer.trusted is False

    restored.strategies[0].trusted = True
    reloaded = pickle.loads(pickle.dumps(restored))
    assert reloaded.strategies[0].trusted is True
    assert reloaded.strategies[0].tool.trusted is False
    assert reloaded.trainer.trusted is False


def test_public_workflow_constructors_default_to_restricted_loading(workflow_cls):
    parameters = inspect.signature(workflow_cls).parameters
    assert parameters["trusted"].default is False


def _entry_points():
    from qlib.contrib.meta.data_selection.dataset import InternalData, MetaDatasetDS
    from qlib.contrib.rolling.ddgda import DDGDA
    from qlib.model.trainer import DelayTrainerR, DelayTrainerRM, end_task_train
    from qlib.workflow import QlibRecorder
    from qlib.workflow.online.strategy import RollingStrategy
    from qlib.workflow.online.update import LabelUpdater, PredUpdater, RMDLoader
    from qlib.workflow.online.utils import OnlineToolR
    from qlib.workflow.recorder import MLflowRecorder, Recorder
    from qlib.workflow.record_temp import RecordTemp

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
        (RecordTemp(None).load, {"name": "unused"}),
        (MLflowRecorder.load_object, {}),
        (Recorder.load_object, {}),
    ]


def test_all_entry_points_use_the_same_public_keyword():
    for entry, _ in _entry_points():
        parameters = inspect.signature(entry).parameters
        assert parameters["trusted"].default is False, entry


@pytest.mark.parametrize("value", ["false", 0, 1, None, np.bool_(True)])
def test_workflow_entries_reject_non_boolean_consent_before_loading(value):
    for entry, kwargs in _entry_points()[:-2]:
        with pytest.raises(TypeError, match="`trusted` must be a bool") as caught:
            entry(**kwargs, trusted=value)
        assert ARTIFACT_MIGRATION_URL in str(caught.value)


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

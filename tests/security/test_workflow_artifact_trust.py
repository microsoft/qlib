from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("qlib.workflow.online.strategy", "RollingStrategy"),
        ("qlib.workflow.online.utils", "OnlineToolR"),
        ("qlib.workflow.online.update", "RMDLoader"),
        ("qlib.workflow.online.update", "DSBasedUpdater"),
        ("qlib.model.trainer", "DelayTrainerR"),
        ("qlib.model.trainer", "DelayTrainerRM"),
        ("qlib.contrib.rolling.ddgda", "DDGDA"),
    ],
)
def test_restored_workflows_without_trust_state_remain_restricted(module_name, class_name):
    if class_name == "DDGDA":
        pytest.importorskip("torch")
    cls = getattr(import_module(module_name), class_name)
    if class_name == "DSBasedUpdater":

        class ConcreteUpdater(cls):
            def get_update_data(self, dataset):
                raise NotImplementedError

        cls = ConcreteUpdater

    legacy = object.__new__(cls)
    opted_in = object.__new__(cls)
    opted_in.trusted_artifacts = True

    assert "trusted_artifacts" not in legacy.__dict__
    assert legacy.trusted_artifacts is False
    assert opted_in.trusted_artifacts is True
    del opted_in.trusted_artifacts
    assert opted_in.trusted_artifacts is False


@pytest.fixture
def online_artifacts(monkeypatch):
    from qlib.workflow.online import update

    dates = pd.date_range("2024-01-01", periods=4)
    index = pd.MultiIndex.from_product([dates, ["SH600000"]], names=["datetime", "instrument"])
    predictions = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0]}, index=index)
    dataset = Mock()
    model = Mock()
    model.predict.return_value = predictions.iloc[2:, 0]
    objects = {
        "pred.pkl": predictions.iloc[:2],
        "label.pkl": predictions.iloc[:2].rename(columns={"score": "LABEL0"}),
        "dataset": dataset,
        "params.pkl": model,
    }
    recorder = Mock()
    recorder.info = {"id": "recording-recorder"}
    recorder.load_object.side_effect = lambda name, **kwargs: objects[name]
    monkeypatch.setattr(update, "D", SimpleNamespace(calendar=lambda **kwargs: dates))
    monkeypatch.setattr(
        update,
        "get_date_by_shift",
        lambda date, shift, **kwargs: pd.Timestamp(date) + pd.Timedelta(days=shift),
    )
    return SimpleNamespace(recorder=recorder, dataset=dataset, model=model, predictions=predictions, dates=dates)


@pytest.mark.parametrize("options", [{}, {"trusted_artifacts": False}, {"trusted_artifacts": True}])
def test_rolling_constructor_propagates_trust_through_online_update(online_artifacts, monkeypatch, options):
    from qlib.workflow.online import strategy

    monkeypatch.setattr(strategy, "TimeAdjuster", Mock())
    rolling = strategy.RollingStrategy("rolling", {}, object.__new__(strategy.RollingGen), **options)
    rolling.tool.online_models = Mock(return_value=[online_artifacts.recorder])

    rolling.tool.update_online_pred(to_date=online_artifacts.dates[-1])

    calls = online_artifacts.recorder.load_object.call_args_list
    assert [item.args[0] for item in calls] == ["pred.pkl", "dataset", "dataset", "params.pkl"]
    assert calls[0].kwargs == {}
    for item in calls[1:]:
        assert item.kwargs.get("trusted", False) is options.get("trusted_artifacts", False)
    online_artifacts.dataset.setup_data.assert_called_once()
    online_artifacts.model.predict.assert_called_once_with(online_artifacts.dataset)
    online_artifacts.recorder.save_objects.assert_called_once()
    pd.testing.assert_frame_equal(
        online_artifacts.recorder.save_objects.call_args.kwargs["pred.pkl"], online_artifacts.predictions
    )


@pytest.mark.parametrize("updater_name", ["PredUpdater", "LabelUpdater"])
@pytest.mark.parametrize("options", [{}, {"trusted_artifacts": False}, {"trusted_artifacts": True}])
def test_updater_preserves_legacy_loader_constructor_until_opt_in(online_artifacts, updater_name, options):
    from qlib.workflow.online import update

    def make_loader(*, rec):
        return SimpleNamespace(rec=rec)

    loader = Mock(side_effect=make_loader)
    updater_cls = getattr(update, updater_name)
    if options.get("trusted_artifacts", False):
        with pytest.raises(TypeError, match="trusted_artifacts"):
            updater_cls(online_artifacts.recorder, loader_cls=loader, **options)
        loader.assert_called_once_with(rec=online_artifacts.recorder, trusted_artifacts=True)
        online_artifacts.recorder.load_object.assert_not_called()
    else:
        updater = updater_cls(online_artifacts.recorder, loader_cls=loader, **options)
        assert updater.rmdl.rec is online_artifacts.recorder
        loader.assert_called_once_with(rec=online_artifacts.recorder)


def test_default_loader_accepts_a_dataset_supplied_by_the_caller():
    from qlib.workflow.online.update import RMDLoader

    recorder = Mock()
    dataset = Mock()
    loader = RMDLoader(recorder)

    assert loader.get_dataset("2024-01-01", "2024-01-02", unprepared_dataset=dataset) is dataset
    recorder.load_object.assert_not_called()
    dataset.config.assert_called_once_with(
        handler_kwargs={"start_time": "2024-01-01", "end_time": "2024-01-02"},
        segments={"test": ("2024-01-01", "2024-01-02")},
    )
    dataset.setup_data.assert_called_once()


@pytest.mark.parametrize("trusted_artifacts", [False, True])
def test_rolling_task_collection_and_generation_honor_strategy_trust(monkeypatch, trusted_artifacts):
    from qlib.workflow.online import strategy

    segment = (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31"))
    task = {"model": {"class": "LinearModel"}, "dataset": {"kwargs": {"segments": {"test": segment}}}}
    recorder = Mock()
    recorder.load_object.return_value = task
    monkeypatch.setattr(strategy, "TimeAdjuster", Mock())
    monkeypatch.setattr(strategy, "transform_end_date", lambda date: date)
    monkeypatch.setattr(strategy, "RecorderCollector", lambda **kwargs: SimpleNamespace(**kwargs))
    rolling_gen = object.__new__(strategy.RollingGen)
    rolling_gen.step = 20
    rolling_gen.gen_following_tasks = Mock(return_value=[task])
    rolling = strategy.RollingStrategy("rolling", task, rolling_gen, trusted_artifacts=trusted_artifacts)
    rolling.tool.online_models = Mock(return_value=[recorder])

    assert rolling.get_collector().rec_key_func(recorder) == ("LinearModel", segment)
    assert rolling._list_latest([recorder]) == ([recorder], segment)
    assert rolling.prepare_tasks(segment[-1]) == [task]

    rolling_gen.gen_following_tasks.assert_called_once_with(task, segment[-1])
    assert recorder.load_object.call_count == 6
    for item in recorder.load_object.call_args_list:
        assert item.args == ("task",)
        assert item.kwargs.get("trusted", False) is trusted_artifacts


@pytest.fixture
def delayed_backend(monkeypatch):
    from qlib.model import trainer

    recorder = Mock()
    recorder.list_tags.return_value = {
        trainer.TrainerR.STATUS_KEY: trainer.TrainerR.STATUS_BEGIN,
        trainer.TrainerRM.TM_ID: "queued-task",
    }
    manager = Mock()
    manager.STATUS_PART_DONE = trainer.TaskManager.STATUS_PART_DONE
    monkeypatch.setattr(trainer, "TaskManager", manager)

    def execute_task(func, task_pool=None, *, experiment_name, **kwargs):
        function_kwargs = {key: value for key, value in kwargs.items() if key not in {"query", "before_status"}}
        return func(recorder, experiment_name, **function_kwargs)

    run_task = Mock(side_effect=execute_task)
    monkeypatch.setattr(trainer, "run_task", run_task)
    return SimpleNamespace(trainer=trainer, recorder=recorder, manager=manager, run_task=run_task)


def _finish_delayed(backend, mode, end_train_func, constructor_options, call_options):
    cls = backend.trainer.DelayTrainerR if mode == "recorder" else backend.trainer.DelayTrainerRM
    delayed = cls("training", end_train_func=end_train_func, **constructor_options)
    if mode == "worker":
        delayed.worker(**call_options)
    else:
        assert delayed.end_train([backend.recorder], **call_options) == [backend.recorder]
        backend.recorder.set_tags.assert_called_once_with(train_status=delayed.STATUS_END)
    if mode == "task-manager":
        backend.manager.return_value.wait.assert_called_once_with(query={"_id": {"$in": ["queued-task"]}})
    return delayed


@pytest.mark.parametrize("mode", ["recorder", "task-manager", "worker"])
@pytest.mark.parametrize(
    "constructor_options,call_options,expected_kwargs",
    [
        ({}, {}, {}),
        ({"trusted_artifacts": False}, {}, {}),
        ({"trusted_artifacts": True}, {}, {"trusted_artifacts": True}),
        ({"trusted_artifacts": True}, {"trusted_artifacts": False}, {"trusted_artifacts": False}),
        ({}, {"trusted_artifacts": True, "marker": "preserved"}, {"trusted_artifacts": True, "marker": "preserved"}),
    ],
    ids=["default", "disabled", "enabled", "disable-override", "enable-override"],
)
def test_delayed_trainers_forward_only_selected_trust(
    delayed_backend, mode, constructor_options, call_options, expected_kwargs
):
    calls = []
    if not expected_kwargs:

        def finish(recorder, experiment_name):
            calls.append((recorder, experiment_name, {}))

    else:

        def finish(recorder, experiment_name, **kwargs):
            calls.append((recorder, experiment_name, kwargs))

    _finish_delayed(delayed_backend, mode, finish, constructor_options, call_options)

    assert calls == [(delayed_backend.recorder, "training", expected_kwargs)]
    if mode == "recorder":
        delayed_backend.run_task.assert_not_called()
    else:
        delayed_backend.run_task.assert_called_once()
        scheduled = delayed_backend.run_task.call_args.kwargs
        assert scheduled["experiment_name"] == "training"
        assert scheduled["before_status"] == delayed_backend.manager.STATUS_PART_DONE
        assert {key: value for key, value in scheduled.items() if key in expected_kwargs} == expected_kwargs
        if not expected_kwargs:
            assert "trusted_artifacts" not in scheduled


@pytest.mark.parametrize("mode", ["recorder", "task-manager", "worker"])
def test_delayed_trainers_allow_per_call_end_function_and_experiment(delayed_backend, mode):
    default_finish = Mock(side_effect=AssertionError("The overridden end function must not run"))
    custom_finish = Mock()

    _finish_delayed(
        delayed_backend,
        mode,
        default_finish,
        {"trusted_artifacts": True},
        {"end_train_func": custom_finish, "experiment_name": "override"},
    )

    default_finish.assert_not_called()
    custom_finish.assert_called_once_with(delayed_backend.recorder, "override", trusted_artifacts=True)


@pytest.mark.parametrize("trusted_artifacts", [False, True])
def test_internal_data_trust_applies_to_tasks_not_predictions(monkeypatch, trusted_artifacts):
    pytest.importorskip("torch")
    from qlib.contrib.meta.data_selection import dataset as meta_dataset

    segment = ("2024-01-01", "2024-01-02")
    task = {"dataset": {"kwargs": {"segments": {"train": segment}}}}
    index = pd.MultiIndex.from_product(
        [pd.date_range(*segment), ["SH600000", "SH600004"]], names=["datetime", "instrument"]
    )
    data = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]}, index=index)
    recorder = Mock()
    recorder.load_object.side_effect = lambda name, **kwargs: {"task": task, "pred.pkl": data}[name]
    handler = Mock()
    handler.dump_all = False
    handler.fetch.return_value = data
    trainer = Mock()
    trainer.has_worker.return_value = False

    def make_trainer(experiment_name):
        assert experiment_name == "internal"
        return trainer

    def run_jobs(jobs):
        return [func(*args, **kwargs) for func, args, kwargs in jobs]

    monkeypatch.setattr(meta_dataset, "init_task_handler", Mock(return_value=handler))
    monkeypatch.setattr(meta_dataset, "RollingGen", Mock())
    monkeypatch.setattr(meta_dataset, "task_generator", Mock(return_value=[task]))
    monkeypatch.setattr(meta_dataset, "R", SimpleNamespace(list_recorders=Mock(return_value={"run": recorder})))
    monkeypatch.setattr(meta_dataset, "Parallel", Mock(return_value=run_jobs))
    calc_perf = Mock(return_value=pd.Series([1.0, 1.0], index=pd.date_range(*segment)))
    monkeypatch.setattr(meta_dataset.InternalData, "_calc_perf", calc_perf)
    internal = meta_dataset.InternalData(task, step=1, exp_name="internal")

    internal.setup(trainer=make_trainer, trusted_artifacts=trusted_artifacts)

    trainer.train.assert_not_called()
    calls = recorder.load_object.call_args_list
    assert len(calls) == 2
    assert calls[0].args == ("pred.pkl",)
    assert calls[0].kwargs == {}
    assert calls[1].args == ("task",)
    assert calls[1].kwargs.get("trusted", False) is trusted_artifacts
    calc_perf.assert_called_once()
    for series in calc_perf.call_args.args:
        pd.testing.assert_series_equal(series, data.iloc[:, 0])
    assert internal.data_ic_df.shape == (2, 1)
    assert internal.data_ic_df.iloc[:, 0].tolist() == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize("options", [{}, {"trusted_artifacts": False}, {"trusted_artifacts": True}])
def test_meta_dataset_forwards_trust_to_internal_data_setup(monkeypatch, options):
    pytest.importorskip("torch")
    from qlib.contrib.meta.data_selection import dataset as meta_dataset

    setup = Mock()
    monkeypatch.setattr(meta_dataset.InternalData, "setup", setup)
    monkeypatch.setattr(meta_dataset.MetaDatasetDS, "_prepare_meta_ipt", Mock(return_value=pd.DataFrame()))
    monkeypatch.setattr(meta_dataset, "MetaTaskDS", Mock())

    dataset = meta_dataset.MetaDatasetDS(
        task_tpl=[{"dataset": {}}], step=1, exp_name="internal", segments=0.5, **options
    )

    assert dataset.internal_data.exp_name == "internal"
    setup.assert_called_once()
    assert setup.call_args.kwargs.get("trusted_artifacts", False) is options.get("trusted_artifacts", False)

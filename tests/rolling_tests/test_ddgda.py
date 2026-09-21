import copy
import pickle

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from joblib import parallel_backend

from qlib.contrib.meta.data_selection.dataset import InternalData
from qlib.contrib.meta.data_selection.model import MetaModelDS, TimeReweighter
from qlib.contrib.rolling.ddgda import DDGDA
from qlib.model.trainer import DelayTrainerR
from qlib.utils import init_instance_by_config
from qlib.utils.exceptions import LoadObjectError
from qlib.workflow import R
from qlib.workflow.online.update import RMDLoader
from qlib.workflow.recorder import UnsafeArtifactWarning


def _task(context):
    dates = context.calendar.strftime("%Y-%m-%d")
    return {
        "model": {
            "class": "LinearModel",
            "module_path": "qlib.contrib.model.linear",
            "kwargs": {"estimator": "ridge", "alpha": 0.05},
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": {
                        "start_time": dates[60],
                        "end_time": dates[219],
                        "fit_start_time": dates[60],
                        "fit_end_time": dates[119],
                        "instruments": context.instruments,
                    },
                },
                "segments": {
                    "train": [dates[60], dates[119]],
                    "valid": [dates[120], dates[139]],
                    "test": [dates[160], dates[199]],
                },
            },
        },
        "record": [
            "qlib.workflow.record_temp.SignalRecord",
            "qlib.workflow.record_temp.SigAnaRecord",
            {
                "class": "PortAnaRecord",
                "module_path": "qlib.workflow.record_temp",
                "kwargs": {
                    "config": {
                        "strategy": {
                            "class": "TopkDropoutStrategy",
                            "module_path": "qlib.contrib.strategy",
                            "kwargs": {"signal": "<PRED>", "topk": 3, "n_drop": 1},
                        },
                        "backtest": {
                            "start_time": dates[160],
                            "end_time": dates[199],
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


@pytest.mark.slow
@pytest.mark.parametrize(
    "workflow_context,sim_task_model", [(8, "linear"), (192, "gbdt")], indirect=["workflow_context"]
)
def test_ddgda_full_workflow(workflow_context, sim_task_model, monkeypatch):
    context = workflow_context
    monkeypatch.chdir(context.root)
    config = context.root / "workflow.yaml"
    config.write_text(yaml.safe_dump({"task": _task(context)}))
    work = context.root / "work"
    work.mkdir()
    workflow = DDGDA(
        conf_path=config,
        exp_name="ddgda-result",
        rolling_exp="ddgda-rolling",
        working_dir="work",
        sim_task_model=sim_task_model,
        train_start=str(context.calendar[60].date()),
        meta_1st_train_end=str(context.calendar[119].date()),
        horizon=1,
        step=20,
        hist_step_n=2,
        loss_skip_thresh=2,
        fea_imp_n=4,
        segments=0.5,
    )
    previous_threads = torch.get_num_threads()
    previous_grad = torch.is_grad_enabled()
    previous_rng = torch.get_rng_state()
    try:
        torch.set_num_threads(1)
        with pytest.raises(pickle.UnpicklingError, match="trusted=True"):
            workflow.run()
        assert not workflow._internal_data_path.exists()
        workflow.trusted = True
        with parallel_backend("threading"), pytest.warns(UnsafeArtifactWarning):
            workflow.run()
    finally:
        torch.set_num_threads(previous_threads)
        torch.set_grad_enabled(previous_grad)
        torch.set_rng_state(previous_rng)

    with pytest.warns(UnsafeArtifactWarning):
        internal_data = workflow._load_cache(workflow._internal_data_path)
    assert isinstance(internal_data, InternalData)
    assert internal_data.data_ic_df.shape == (160, 7)
    assert internal_data.data_ic_df.notna().any().all()
    similarity_recorders = R.list_recorders(experiment_name=internal_data.exp_name)
    assert len(similarity_recorders) == 7
    if sim_task_model == "gbdt":
        for similarity_recorder in similarity_recorders.values():
            with pytest.warns(UnsafeArtifactWarning):
                similarity_model = RMDLoader(similarity_recorder, trusted=True).get_model()
            assert similarity_model.early_stopping_rounds is None
            assert similarity_model.num_boost_round == 150
            assert similarity_model.model.num_trees() > 1
    assert (work / "handler_proxy.pkl").is_file()
    assert (work / "fea_label_df.pkl").is_file()
    assert workflow._task_path.is_file()

    meta_recorders = R.list_recorders(experiment_name=workflow.meta_exp_name)
    assert len(meta_recorders) == 1
    meta_recorder = next(iter(meta_recorders.values()))
    with pytest.raises(LoadObjectError, match="MetaModelDS"):
        meta_recorder.load_object("model")
    with pytest.warns(UnsafeArtifactWarning):
        meta_model = meta_recorder.load_object("model", trusted=True)
    assert isinstance(meta_model, MetaModelDS)
    assert meta_model.fitted
    assert meta_model.max_epoch == 30
    for name in ("loss/train", "loss/test", "ic/train", "ic/test"):
        assert np.isfinite(meta_recorder.list_metrics()[name])
    history = meta_recorder.client.get_metric_history(meta_recorder.id, "loss/train")
    assert {entry.step for entry in history} == set(range(30))
    assert np.isfinite([entry.value for entry in history]).all()
    assert np.ptp([entry.value for entry in history]) > 1e-8
    assert all(torch.isfinite(parameter).all() for parameter in meta_model.tn.parameters())

    records = R.list_recorders(experiment_name=workflow.rolling_exp)
    assert len(records) == 2
    rolling_predictions = {}
    for recorder in records.values():
        with pytest.raises(LoadObjectError, match="LinearModel"):
            RMDLoader(recorder).get_model()
        with pytest.raises(LoadObjectError, match="DatasetH"):
            recorder.load_object("dataset")
        with pytest.raises(LoadObjectError, match="Forbidden class"):
            recorder.load_object("task")
        with pytest.warns(UnsafeArtifactWarning):
            task = recorder.load_object("task", trusted=True)
            model = RMDLoader(recorder, trusted=True).get_model()
            dataset = recorder.load_object("dataset", trusted=True)
        assert isinstance(task["reweighter"], TimeReweighter)
        assert np.isfinite(task["reweighter"].time_weight).all()
        assert (task["reweighter"].time_weight > 0).all()
        assert task["reweighter"].time_weight.std() > 0
        dataset.setup_data(handler_kwargs={"init_type": "load_state"})
        actual = model.predict(dataset)
        expected = recorder.load_object("pred.pkl").iloc[:, 0]
        pd.testing.assert_series_equal(actual, expected, check_names=False, check_exact=True)
        rolling_predictions[tuple(task["dataset"]["kwargs"]["segments"]["test"])] = expected

    assert workflow._task_path.stat().st_size < 10000
    with pytest.warns(UnsafeArtifactWarning):
        saved_tasks = workflow._load_cache(workflow._task_path)
    assert len(saved_tasks) == 2
    assert all(isinstance(task["dataset"]["kwargs"]["handler"], dict) for task in saved_tasks)
    replay_trainer = DelayTrainerR(experiment_name="ddgda-replay")
    replay_records = replay_trainer.train(saved_tasks)
    with pytest.raises(LoadObjectError, match="TimeReweighter"):
        replay_trainer.end_train(replay_records)
    with pytest.warns(UnsafeArtifactWarning):
        replay_trainer.end_train(replay_records, trusted=True)
    for task, replay_recorder in zip(saved_tasks, replay_records):
        replay_prediction = replay_recorder.load_object("pred.pkl").iloc[:, 0]
        expected = rolling_predictions[tuple(task["dataset"]["kwargs"]["segments"]["test"])]
        pd.testing.assert_series_equal(replay_prediction, expected, check_exact=True)

    recorder = R.get_recorder(experiment_name=workflow.exp_name, recorder_id=workflow._rid)
    predictions = recorder.load_object("pred.pkl")
    labels = recorder.load_object("label.pkl")
    expected_index = pd.MultiIndex.from_product(
        [context.calendar[160:200], context.instruments], names=["datetime", "instrument"]
    )
    pd.testing.assert_index_equal(predictions.index, expected_index)
    pd.testing.assert_index_equal(labels.index, expected_index)
    assert np.isfinite(predictions.values).all()
    assert np.isfinite(labels.values).all()
    ic = recorder.load_object("sig_analysis/ic.pkl")
    assert len(ic) == 40
    assert np.isfinite(ic).all()
    report = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    pd.testing.assert_index_equal(report.index, context.calendar[160:200].rename("datetime"), check_names=False)
    assert np.isfinite(report[["return", "cost", "bench", "account"]].values).all()
    assert report["cost"].sum() > 0
    assert report["turnover"].sum() > 0


def test_internal_data_daily_rank_ic():
    dates = pd.DatetimeIndex(["2020-01-03", "2020-01-06", "2020-01-07"], name="datetime")
    index = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["datetime", "instrument"])
    pred = pd.Series([1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0], index=index)
    label = pd.Series([1.0, 2.0, 3.0] * 3, index=index)
    result = InternalData({}, 20, "unused")._calc_perf(pred, label)
    pd.testing.assert_series_equal(result, pd.Series([1.0, -1.0, np.nan], index=dates, name="label"))


def test_ddgda_cache_requires_explicit_trust(tmp_path):
    path = tmp_path / "internal.pkl"
    expected = InternalData({}, 20, "unused")
    path.write_bytes(pickle.dumps(expected))
    workflow = object.__new__(DDGDA)
    with pytest.raises(pickle.UnpicklingError, match="trusted=True"):
        workflow._load_cache(path)
    workflow.trusted = True
    with pytest.warns(UnsafeArtifactWarning, match="cache source and storage"):
        actual = workflow._load_cache(path)
    assert isinstance(actual, InternalData)
    assert actual.__dict__ == expected.__dict__
    workflow.trusted = False
    with pytest.raises(pickle.UnpicklingError, match="trusted=True"):
        workflow._load_cache(path)


def test_ddgda_external_cache_keeps_a_reloadable_reference(tmp_path):
    path = tmp_path / "handler.pkl"
    expected = {"data": [1, 2, 3]}
    path.write_bytes(pickle.dumps(expected))
    workflow = DDGDA(conf_path=tmp_path / "unused.yaml", h_path=path, trusted=True)
    task = {"dataset": {"kwargs": {"handler": "replaced-by-h-path"}}}
    task = workflow._replace_handler_with_cache(task)
    for model_type in ("linear", "gbdt"):
        workflow._adjust_task(task, model_type)
        task = workflow._replace_handler_with_cache(task, tmp_path / "unused")
        handler = task["dataset"]["kwargs"]["handler"]
        assert handler["kwargs"] == {"path": str(path), "trusted": True}
        with pytest.warns(UnsafeArtifactWarning):
            assert init_instance_by_config(handler) == expected
    assert not (tmp_path / "unused").exists()
    workflow.trusted = False
    task = workflow._replace_handler_with_cache(task)
    assert task["dataset"]["kwargs"]["handler"]["kwargs"]["trusted"] is False
    assert init_instance_by_config(task["dataset"]["kwargs"]["handler"]) == expected


@pytest.mark.parametrize("value", ["false", 0, 1, None, np.bool_(True)])
def test_ddgda_cache_rejects_non_boolean_trust(tmp_path, value):
    workflow = object.__new__(DDGDA)
    workflow.trusted = value
    with pytest.raises(TypeError, match="must be a bool"):
        workflow._load_cache(tmp_path / "not-opened.pkl")


def test_ddgda_adjusted_task_does_not_mutate_defaults():
    from qlib.contrib.rolling.ddgda import LGBM_MODEL, LINEAR_MODEL, PROC_ARGS

    workflow = object.__new__(DDGDA)
    expected = copy.deepcopy((LGBM_MODEL, LINEAR_MODEL, PROC_ARGS))
    for kind in ("gbdt", "linear"):
        task = {"dataset": {"kwargs": {"handler": {"kwargs": {}}}}}
        workflow._adjust_task(task, kind)
        task["model"]["kwargs"]["num_boost_round"] = 150
        if kind == "linear":
            task["dataset"]["kwargs"]["handler"]["kwargs"]["infer_processors"][0]["kwargs"]["clip_outlier"] = False
        task["dataset"]["kwargs"]["handler"]["kwargs"].clear()
    assert (LGBM_MODEL, LINEAR_MODEL, PROC_ARGS) == expected

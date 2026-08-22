from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from examples.courage_strict_continuous_v1.run_diagnostic_v1 import (
    validate_contract as validate_diagnostic_contract,
)
from qlib.contrib.data.courage_strict_v1.dataset import CourageStrictV1Dataset
from qlib.contrib.model.courage_strict_continuous_v1 import (
    StratifiedTimeCellDistributedSampler,
    _moment_summary,
    _pointwise_loss,
    _raw_moment_totals,
    continuous_scheduler,
    effective_target_mask,
    load_config,
    optimizer_parameter_groups,
    validate_identity,
)
from qlib.contrib.model.patchtst_courage_strict_v1 import PatchTSTCourageStrictV1

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "examples/courage_strict_continuous_v1/config.json"
PIT_10_15_CONFIG = (
    ROOT / "examples/courage_strict_continuous_v1/config_pit_10_15_v1.json"
)
MSE_CONFIG = ROOT / (
    "examples/courage_strict_continuous_v1/"
    "config_pit_10_15_source_semantics_v2_mse_v1.json"
)
DIAGNOSTIC_CONFIG = (
    ROOT / "examples/courage_strict_continuous_v1/config_diagnostic_v1.json"
)


def test_config_and_p0_identity_are_closed() -> None:
    config = load_config(CONFIG)
    assert validate_identity(config) == config.source["identity"]
    assert config.source["model"]["freeze_backbone"] is False
    assert config.source["training"]["global_batch_size"] == 1024


def test_pit_10_15_config_adds_only_the_frozen_candidate_identity() -> None:
    config = load_config(PIT_10_15_CONFIG)
    assert validate_identity(config) == config.source["identity"]
    assert config.source["membership_candidate"]["turnover_band"] == [10.0, 15.0]
    assert config.source["membership_candidate"]["only_changed_training_input"] is True
    assert config.source["scaler_policy"] == (
        "reuse_baseline_train_scaler_fixed_preprocessing"
    )
    assert config.source["training"]["validation_interval_steps"] == 50
    assert "run_eval50_v1" in str(config.output_root)


def test_mse_config_changes_only_the_declared_loss_contract() -> None:
    config = load_config(MSE_CONFIG)
    assert validate_identity(config) == config.source["identity"]
    assert config.source["training"]["loss_function"] == "MSE"
    assert config.source["selection"]["metric"] == ("valid_equal_head_standardized_mse")


def test_diagnostic_config_preserves_original_population_and_parameters() -> None:
    assert validate_diagnostic_contract()["authority"]["training"] is True
    base = load_config(CONFIG)
    diagnostic = load_config(DIAGNOSTIC_CONFIG)
    assert validate_identity(diagnostic) == diagnostic.source["identity"]
    assert diagnostic.provider_root == base.provider_root
    assert diagnostic.origins == base.origins
    assert diagnostic.source["training"] == base.source["training"]
    assert diagnostic.source["model"] == base.source["model"]
    assert diagnostic.source["diagnostics"]["enabled"] is True
    assert diagnostic.output_root != base.output_root


def test_raw_prediction_moment_summary_is_exact() -> None:
    prediction = torch.tensor([[-1.0] * 7, [1.0] * 7])
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    totals = _raw_moment_totals(
        prediction,
        target,
        mask,
        target_medians=torch.zeros(7, dtype=torch.float64),
        target_iqrs=torch.ones(7, dtype=torch.float64),
    )
    summary = _moment_summary(totals)
    for horizon in (5, 15, 30, 60, 120, 240, 480):
        row = summary[str(horizon)]
        assert row["rows"] == 2
        assert row["prediction_mean"] == pytest.approx(0.0)
        assert row["prediction_std"] == pytest.approx(1.0)
        assert row["target_mean"] == pytest.approx(0.0)
        assert row["target_std"] == pytest.approx(0.0)
        assert row["bias"] == pytest.approx(0.0)
        assert row["prediction_up_rate"] == pytest.approx(0.5)
        assert row["target_up_rate"] == pytest.approx(0.0)
        assert row["prediction_zero_rate"] == pytest.approx(0.0)
        assert row["target_zero_rate"] == pytest.approx(1.0)
    prediction = torch.tensor([[2.0, -1.0]])
    target = torch.tensor([[0.0, 1.0]])
    assert torch.equal(
        _pointwise_loss(prediction, target, loss_function="MSE"),
        torch.tensor([[4.0, 4.0]]),
    )


def test_pit_10_15_dataset_references_obey_the_sub_band() -> None:
    dataset = CourageStrictV1Dataset(
        provider_root=ROOT / "data/courage_strict_v1/qlib_provider",
        segments={"selected": ("2026-01-05", "2026-01-06")},
        turnover_band=(10.0, 15.0),
    ).prepare("selected", cache_capacity=1)
    assert len(dataset) > 0
    for position in np.unique(dataset.symbol_positions):
        selected = dataset.symbol_positions == position
        values = dataset.provider.array(int(position), "turnover_mean_60")[
            dataset.ends[selected]
        ]
        assert np.isfinite(values).all()
        assert ((values >= 10.0) & (values <= 15.0)).all()


def test_effective_target_mask_uses_session_local_phase() -> None:
    slots = torch.tensor([0, 5, 15, 30, 121, 126, 136, 151])
    session = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    batch = {
        "target_mask": torch.ones((len(slots), 7), dtype=torch.bool),
        "minute_index": slots[:, None],
        "session_index": session[:, None],
    }
    mask = effective_target_mask(batch)
    assert mask[:, :4].all()
    assert mask[:, 4].tolist() == [True, False, True, True, True, False, True, True]
    assert mask[:, 5].tolist() == [True, False, False, True, True, False, False, True]
    assert torch.equal(mask[:, 5], mask[:, 6])


def test_stratified_sampler_is_complete_disjoint_and_deterministic() -> None:
    ends = np.repeat(np.array([10, 20, 30, 40]), [3, 2, 4, 3])
    samplers = [
        StratifiedTimeCellDistributedSampler(ends, num_replicas=2, rank=rank, seed=17)
        for rank in range(2)
    ]
    left, right = [list(item) for item in samplers]
    assert len(left) == len(right) == 6
    assert set(left).isdisjoint(right)
    assert set(left + right) == set(range(len(ends)))
    assert left == list(samplers[0])
    samplers[0].set_epoch(1)
    assert left != list(samplers[0])


def test_sampler_resume_offset_is_exact() -> None:
    ends = np.repeat(np.array([10, 20, 30, 40]), 4)
    sampler = StratifiedTimeCellDistributedSampler(
        ends, num_replicas=2, rank=0, seed=23
    )
    full = list(sampler)
    sampler.set_start_offset(3)
    assert list(sampler) == full[3:]
    assert len(sampler) == len(full) - 3


def test_optimizer_excludes_bias_norm_and_embedding_from_decay() -> None:
    model = PatchTSTCourageStrictV1(industry_vocab_size=150)
    groups = optimizer_parameter_groups(model, weight_decay=0.01)
    assert groups[0]["weight_decay"] == pytest.approx(0.01)
    assert groups[1]["weight_decay"] == 0.0
    assert sum(len(group["params"]) for group in groups) == len(
        list(model.parameters())
    )


def test_scheduler_is_single_warmup_then_decay() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = continuous_scheduler(optimizer, total_steps=100, warmup_fraction=0.05)
    values = []
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        values.append(optimizer.param_groups[0]["lr"])
    assert values[0] < values[3] <= 1.0
    assert values[4] >= values[5]
    assert values[-1] == pytest.approx(0.0, abs=1e-12)

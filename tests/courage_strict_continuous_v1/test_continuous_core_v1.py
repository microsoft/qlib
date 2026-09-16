from __future__ import annotations

import numpy as np
import pytest
import torch

from qlib.contrib.model.courage_strict_continuous_v1 import (
    StratifiedTimeCellDistributedSampler,
    continuous_scheduler,
    effective_target_mask,
    optimizer_parameter_groups,
)
from qlib.contrib.model.patchtst_courage_strict_v1 import PatchTSTCourageStrictV1


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


def test_optimizer_and_scheduler_contract() -> None:
    model = PatchTSTCourageStrictV1(industry_vocab_size=150)
    groups = optimizer_parameter_groups(model, weight_decay=0.01)
    assert groups[0]["weight_decay"] == pytest.approx(0.01)
    assert groups[1]["weight_decay"] == 0.0
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

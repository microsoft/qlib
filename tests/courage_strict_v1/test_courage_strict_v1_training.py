from __future__ import annotations

import torch

from qlib.contrib.model.courage_strict_v1 import (
    CourageStrictV1Model,
    CourageStrictV1TrainingError,
    _equal_head_loss,
)


def test_equal_head_huber_uses_only_active_cells() -> None:
    prediction = torch.tensor([[0.0, 2.0], [2.0, 100.0]], requires_grad=True)
    target = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    mask = torch.tensor([[True, True], [True, False]])
    loss = _equal_head_loss(prediction, target, mask, world_size=1)
    # Head 0: (0 + 1.5) / 2; head 1: 1.5 / 1; equal-head mean.
    assert torch.isclose(loss, torch.tensor(1.125))
    loss.backward()
    assert prediction.grad is not None
    assert prediction.grad[1, 1] == 0


def test_model_epoch_budget_fails_closed() -> None:
    try:
        CourageStrictV1Model(min_epochs=3, max_epochs=2)
    except CourageStrictV1TrainingError:
        return
    raise AssertionError("invalid epoch budget was accepted")


def test_model_is_a_qlib_model() -> None:
    model = CourageStrictV1Model(num_workers=0)
    assert model.batch_size == 256
    assert model.fitted is False

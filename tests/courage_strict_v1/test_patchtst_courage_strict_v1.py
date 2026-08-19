from __future__ import annotations

import pytest
import torch

from qlib.contrib.model.patchtst_courage_strict_v1 import (
    CourageStrictModelError,
    PatchTSTCourageStrictV1,
)


def _inputs(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "values": torch.zeros(batch, 1200, 12),
        "available": torch.ones(batch, 1200, 12, dtype=torch.bool),
        "missing": torch.zeros(batch, 1200, 12, dtype=torch.bool),
        "padding": torch.zeros(batch, 1200, dtype=torch.bool),
        "minute_index": torch.arange(1200).remainder(240).repeat(batch, 1),
        "session_index": torch.arange(1200)
        .remainder(240)
        .ge(120)
        .long()
        .repeat(batch, 1),
        "slow_values": torch.zeros(batch, 5),
        "slow_available": torch.ones(batch, 5, dtype=torch.bool),
        "industry_id": torch.ones(batch, dtype=torch.long),
    }


def test_strict_model_outputs_seven_return_heads() -> None:
    model = PatchTSTCourageStrictV1(industry_vocab_size=8)
    model.eval()
    with torch.no_grad():
        output = model(**_inputs())
    assert output.shape == (2, 7)
    assert model.parameter_count < 1_200_000


def test_strict_model_rejects_unavailable_nonzero_value() -> None:
    model = PatchTSTCourageStrictV1(industry_vocab_size=8)
    values = _inputs(batch=1)
    values["available"][0, 0, 0] = False
    values["values"][0, 0, 0] = 1.0
    with pytest.raises(CourageStrictModelError, match="invariant"):
        model(**values)


def test_strict_model_rejects_non_main_contract_width() -> None:
    model = PatchTSTCourageStrictV1(industry_vocab_size=8)
    values = _inputs(batch=1)
    values["values"] = torch.zeros(1, 1200, 13)
    with pytest.raises(CourageStrictModelError, match="shape"):
        model(**values)

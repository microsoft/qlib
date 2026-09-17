# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
import runpy
from unittest.mock import Mock, call

import pytest
from ruamel.yaml import YAML

import qlib


TRA_DIR = Path(__file__).resolve().parents[1] / "examples" / "benchmarks" / "TRA"


@pytest.mark.parametrize(
    "config_name",
    [
        "config_alstm.yaml",
        "config_alstm_tra.yaml",
        "config_alstm_tra_init.yaml",
        "config_transformer.yaml",
        "config_transformer_tra.yaml",
        "config_transformer_tra_init.yaml",
    ],
)
def test_tra_example_forwards_initialization_config(monkeypatch, config_name):
    monkeypatch.chdir(TRA_DIR)
    config_path = Path("configs") / config_name
    with config_path.open() as stream:
        config = YAML(typ="safe", pure=True).load(stream)

    main = runpy.run_path(str(TRA_DIR / "example.py"))["main"]
    initialize = Mock()
    dataset, model = Mock(), Mock()
    factory = Mock(side_effect=[dataset, model])
    monkeypatch.setattr(qlib, "init", initialize)
    monkeypatch.setitem(main.__globals__, "init_instance_by_config", factory)

    main(seed=42, config_file=str(config_path))

    initialize.assert_called_once_with(**config["qlib_init"])
    config["task"]["model"]["kwargs"]["seed"] = 42
    assert factory.call_args_list == [call(config["task"]["dataset"]), call(config["task"]["model"])]
    model.fit.assert_called_once_with(dataset)

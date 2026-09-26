# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))
from data_collector.yahoo import collector


@pytest.fixture
def update_env(tmp_path, monkeypatch):
    # BaseRun expects the command-line module to be importable as "collector".
    monkeypatch.setitem(sys.modules, "collector", collector)
    runner = collector.Run(
        source_dir=tmp_path / "source",
        normalize_dir=tmp_path / "normalize",
        max_workers=2,
        region="us",
    )
    data_dir = tmp_path / "qlib_data"
    calendar_dir = data_dir / "calendars"
    calendar_dir.mkdir(parents=True)
    (calendar_dir / "day.txt").write_text("2022-01-03\n2022-01-04\n", encoding="utf-8")

    download = Mock()
    monkeypatch.setattr(collector.GetData, "qlib_data", download)
    monkeypatch.setattr(runner, "download_data", Mock())
    monkeypatch.setattr(runner, "normalize_data_1d_extend", Mock())
    monkeypatch.setattr(collector, "DumpDataUpdate", Mock())
    index_module = ModuleType("data_collector.us_index.collector")
    index_module.get_instruments = Mock()
    monkeypatch.setitem(sys.modules, index_module.__name__, index_module)
    return runner, data_dir, download


@pytest.mark.parametrize("exists_skip", [False, True])
@pytest.mark.parametrize("delete_old", [None, False, True], ids=["default", "keep", "delete"])
def test_update_data_to_bin_forwards_download_options(update_env, monkeypatch, exists_skip, delete_old):
    runner, data_dir, download = update_env
    monkeypatch.setattr(collector, "exists_qlib_data", lambda _: False)
    kwargs = {} if delete_old is None else {"delete_old": delete_old}

    runner.update_data_to_bin(data_dir, end_date="2022-01-05", exists_skip=exists_skip, **kwargs)

    download.assert_called_once_with(
        target_dir=str(data_dir.resolve()),
        interval="1d",
        region="us",
        exists_skip=exists_skip,
        delete_old=True if delete_old is None else delete_old,
    )
    runner.download_data.assert_called_once_with(delay=1, start="2022-01-03", end="2022-01-05", check_data_length=None)
    runner.normalize_data_1d_extend.assert_called_once_with(str(data_dir.resolve()))
    collector.DumpDataUpdate.return_value.dump.assert_called_once_with()


@pytest.mark.parametrize("delete_old", [False, True])
def test_update_data_to_bin_cli_delete_old(update_env, monkeypatch, delete_old):
    runner, data_dir, download = update_env
    monkeypatch.setattr(collector, "exists_qlib_data", lambda _: False)

    collector.fire.Fire(
        runner,
        command=[
            "update_data_to_bin",
            "--qlib_data_1d_dir",
            str(data_dir),
            "--end_date",
            "2022-01-05",
            "--delete_old",
            str(delete_old),
            "--exists_skip",
            "True",
        ],
    )

    download.assert_called_once_with(
        target_dir=str(data_dir.resolve()), interval="1d", region="us", exists_skip=True, delete_old=delete_old
    )


@pytest.mark.parametrize("exists_skip", [False, True])
@pytest.mark.parametrize("delete_old", [False, True])
def test_update_data_to_bin_skips_download_for_existing_data(update_env, monkeypatch, exists_skip, delete_old):
    runner, data_dir, download = update_env
    monkeypatch.setattr(collector, "exists_qlib_data", lambda _: True)

    runner.update_data_to_bin(data_dir, end_date="2022-01-05", exists_skip=exists_skip, delete_old=delete_old)

    download.assert_not_called()
    runner.download_data.assert_called_once()
    collector.DumpDataUpdate.return_value.dump.assert_called_once_with()

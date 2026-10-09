# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import subprocess
from pathlib import Path
from unittest import mock

from qlib.workflow.expm import _file_uri_to_path
from qlib.workflow.recorder import MLflowRecorder


def _make_recorder() -> MLflowRecorder:
    recorder = MLflowRecorder.__new__(MLflowRecorder)
    recorder.id = "run"
    recorder.client = mock.Mock()
    return recorder


def test_file_uri_lock_path_stays_absolute(tmp_path: Path) -> None:
    mlruns = (tmp_path / "mlruns").resolve()

    assert _file_uri_to_path(mlruns.as_uri()) == mlruns
    assert _file_uri_to_path("file:" + str(mlruns)) == mlruns


def test_log_uncommitted_code_skips_non_git_cwd() -> None:
    recorder = _make_recorder()
    not_git = subprocess.CalledProcessError(128, ["git", "rev-parse"])

    with (
        mock.patch(
            "qlib.workflow.recorder.subprocess.run",
            side_effect=not_git,
        ) as run,
        mock.patch(
            "qlib.workflow.recorder.subprocess.check_output"
        ) as check_output,
    ):
        recorder._log_uncommitted_code()

    run.assert_called_once_with(
        ["git", "rev-parse", "--is-inside-work-tree"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    check_output.assert_not_called()
    recorder.client.log_text.assert_not_called()


def test_log_uncommitted_code_uses_git_without_shell() -> None:
    recorder = _make_recorder()
    in_worktree = subprocess.CompletedProcess(["git"], 0, stdout=b"true\n")

    with (
        mock.patch(
            "qlib.workflow.recorder.subprocess.run",
            return_value=in_worktree,
        ),
        mock.patch(
            "qlib.workflow.recorder.subprocess.check_output",
            side_effect=[b"diff", b"status", b"cached"],
        ) as check_output,
    ):
        recorder._log_uncommitted_code()

    assert [call.args[0] for call in check_output.call_args_list] == [
        ["git", "diff"],
        ["git", "status"],
        ["git", "diff", "--cached"],
    ]
    assert all(
        call.kwargs == {"stderr": subprocess.DEVNULL}
        for call in check_output.call_args_list
    )
    assert recorder.client.log_text.call_count == 3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import subprocess
from unittest.mock import MagicMock

from qlib.workflow.expm import _file_uri_to_path
from qlib.workflow.recorder import MLflowRecorder


def test_file_uri_path_remains_absolute_from_other_cwd(tmp_path, monkeypatch):
    target = tmp_path / "mlruns"
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    file_uri = target.resolve().as_uri()

    assert _file_uri_to_path(file_uri) == target.resolve()
    assert _file_uri_to_path(file_uri).is_absolute()


def test_log_uncommitted_code_skips_non_git_cwd(monkeypatch, tmp_path):
    recorder = MLflowRecorder.__new__(MLflowRecorder)
    recorder.client = MagicMock()
    recorder.id = "run-id"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("qlib.workflow.recorder._is_git_work_tree", lambda: False)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("git diff commands should not run outside a git work tree")

    monkeypatch.setattr(subprocess, "run", fail_if_called)

    recorder._log_uncommitted_code()

    recorder.client.log_text.assert_not_called()


def test_log_uncommitted_code_captures_git_output_without_shell(monkeypatch):
    recorder = MLflowRecorder.__new__(MLflowRecorder)
    recorder.client = MagicMock()
    recorder.id = "run-id"
    monkeypatch.setattr("qlib.workflow.recorder._is_git_work_tree", lambda: True)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout=b"captured", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    recorder._log_uncommitted_code()

    assert [cmd for cmd, _ in calls] == [
        ["git", "diff"],
        ["git", "status"],
        ["git", "diff", "--cached"],
    ]
    for _, kwargs in calls:
        assert kwargs["check"] is True
        assert kwargs["stdout"] is subprocess.PIPE
        assert kwargs["stderr"] is subprocess.PIPE
    assert recorder.client.log_text.call_count == 3

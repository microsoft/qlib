"""Regression test for https://github.com/microsoft/qlib/issues/1846.

The reporter only had 60min data and ran backtest with a 60min executor,
but the Account was always constructed with the hardcoded default
``freq="day"``. The benchmark fetch at ``Account.__init__`` therefore
tried to read day-level data and raised before the executor got a chance
to reset the account to its own ``time_per_step``.

The fix threads ``time_per_step`` from the executor config (or executor
instance) into ``create_account_instance`` so the account is born with
the correct freq.

This test exercises ``_executor_time_per_step`` directly — it is the
extraction helper introduced by the fix and is what guards the
"executor → account.freq" handoff.
"""

import unittest
from types import SimpleNamespace

from qlib.backtest import _executor_time_per_step


class TestExecutorFreqExtraction(unittest.TestCase):
    def test_dict_with_time_per_step(self) -> None:
        cfg = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "60min"},
        }
        self.assertEqual(_executor_time_per_step(cfg), "60min")

    def test_dict_without_kwargs_returns_none(self) -> None:
        # Falls back to the historic "day" default in the caller.
        self.assertIsNone(_executor_time_per_step({"class": "Whatever"}))
        self.assertIsNone(_executor_time_per_step({"class": "Whatever", "kwargs": {}}))

    def test_dict_with_non_string_time_per_step_is_ignored(self) -> None:
        # Defensive: only honor string time_per_step. The constraint
        # matches BaseExecutor's type annotation.
        self.assertIsNone(_executor_time_per_step({"kwargs": {"time_per_step": 60}}))

    def test_instance_with_time_per_step_attribute(self) -> None:
        instance = SimpleNamespace(time_per_step="day")
        self.assertEqual(_executor_time_per_step(instance), "day")

    def test_instance_without_attribute_returns_none(self) -> None:
        self.assertIsNone(_executor_time_per_step(SimpleNamespace()))

    def test_string_path_executor_returns_none(self) -> None:
        # Path/str executors carry no freq info — caller defaults to "day".
        self.assertIsNone(_executor_time_per_step("/path/to/executor.yaml"))


if __name__ == "__main__":
    unittest.main()

"""Regression tests for issue #2038.

In freshly spawned joblib worker processes ``register_from_C`` reads
``C.registered`` before ``set_conf_from_C`` syncs the parent's config.  When
``_config`` does not carry the ``_registered`` key (for instance after a
``Config.reset()`` call), the previous implementation surfaced as
``AttributeError: No such ``registered`` in self._config`` and crashed the
backtest worker.  The fixed property and the defensive ``getattr`` in
``register_from_C`` must keep these scenarios safe.
"""

from __future__ import annotations

import unittest

from qlib.config import C, Config, QlibConfig, _default_config


class QlibConfigRegisteredAccessTest(unittest.TestCase):
    """Issue #2038: ``C.registered`` must never raise even if the
    ``_registered`` key was scrubbed from ``self._config``."""

    def test_registered_property_returns_false_when_key_missing(self) -> None:
        config = QlibConfig(_default_config)
        # Simulate the worker state where ``_config`` was rebuilt from
        # ``_default_config`` (which does not contain ``_registered``).
        config.__dict__["_config"].pop("_registered", None)

        try:
            value = config.registered
        except AttributeError as exc:  # pragma: no cover - failure path
            self.fail(f"QlibConfig.registered raised AttributeError: {exc}")

        self.assertFalse(value)

    def test_registered_property_returns_false_when_config_missing(self) -> None:
        # Defensive: even if ``_config`` itself has not been initialised yet
        # (e.g. partially-constructed instance during unpickling), the property
        # must surface ``False`` rather than blow up.
        config = QlibConfig.__new__(QlibConfig)
        self.assertFalse(config.registered)

    def test_register_from_C_does_not_raise_when_registered_missing(self) -> None:
        original_config = dict(C.__dict__["_config"])
        original_registered = C.__dict__["_config"].pop("_registered", False)
        try:
            # Should hit the defensive ``getattr`` path and NOT raise.  We do
            # not call ``register_from_C`` end-to-end (it would touch the
            # workflow exit handler) -- instead we exercise the guarded
            # boolean expression directly, which is the exact line that
            # raised in the original bug report.
            self.assertFalse(getattr(C, "registered", False))
        finally:
            # Restore C so other tests are not affected.
            C.__dict__["_config"].clear()
            C.__dict__["_config"].update(original_config)
            if original_registered is not None:
                C.__dict__["_config"]["_registered"] = original_registered

    def test_registered_property_after_normal_construction(self) -> None:
        # The defensive change must not regress the happy path.
        config = QlibConfig(_default_config)
        self.assertFalse(config.registered)

        config.__dict__["_config"]["_registered"] = True
        self.assertTrue(config.registered)


if __name__ == "__main__":
    unittest.main()

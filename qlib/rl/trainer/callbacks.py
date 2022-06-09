# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Callbacks to insert customized recipes during the training.
Mimicks the hooks of Keras / PyTorch-Lightning, but tailored for the context of RL.
"""

from __future__ import annotations

import logging

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from qlib.typehint import Literal

if TYPE_CHECKING:
    from .trainer import Trainer
    from .vessel import TrainingVesselBase


_logger = logging.getLogger(__name__)

class Callback:
    """Base class of all callbacks."""

    def on_fit_start(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """Called before the whole fit process begins."""

    def on_fit_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """Called after the whole fit process ends."""

    def on_train_start(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """Called when each collect for training begins."""

    def on_train_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """Called when the training ends.
        To access all outputs produced during training, cache the data in either trainer and vessel,
        and post-process them in this hook.
        """

    def on_validate_start(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """Called when every run for validation begins."""

    def on_validate_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """Called when the validation ends."""

    def on_test_start(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """Called when every run of testing begins."""

    def on_test_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """Called when the testing ends."""


class EarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving.

    Reference: https://github.com/keras-team/keras/blob/v2.9.0/keras/callbacks.py#L1744-L1893

    Assuming the goal of a training is to minimize the loss. With this, the
  metric to be monitored would be `'loss'`, and mode would be `'min'`. A
  `model.fit()` training loop will check at end of every epoch whether
  the loss is no longer decreasing, considering the `min_delta` and
  `patience` if applicable. Once it's found no longer decreasing,
  `model.stop_training` is marked True and the training terminates.
  The quantity to be monitored needs to be available in `logs` dict.
  To make it so, pass the loss or metrics at `model.compile()`.
    """
    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.,
        patience: int = 0,
        mode: Literal["auto", "min", "max"] = "auto",
        baseline: float | None = None,
        restore_best_weights: bool = False,
    ):
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ["auto", "min", "max"]:
            raise ValueError("Unsupported earlystopping mode: " + mode)

        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            if self.monitor.endswith("acc") or self.monitor.endswith("accuracy") or self.monitor.endswith("auc"):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.inf if self.monitor_op == np.less else -np.inf
        self.best_weights = None
        self.best_epoch = 0

    def on_validate_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = self.model.get_weights()

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            # Only restart wait if we beat both the baseline and our previous best.
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0

        # Only check after the first epoch.
        if self.wait >= self.patience and epoch > 0:
            trainer.should_stop = True
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    io_utils.print_msg(
                        "Restoring model weights from the end of the best epoch: " f"{self.best_epoch + 1}."
                    )
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            io_utils.print_msg(f"Epoch {self.stopped_epoch + 1}: early stopping")

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(
                "Early stopping conditioned on metric `%s` " "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)

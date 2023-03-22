# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Callbacks to insert customized recipes during the training.
Mimicks the hooks of Keras / PyTorch-Lightning, but tailored for the context of RL.
"""

from __future__ import annotations

import copy
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from qlib.log import get_module_logger
from qlib.typehint import Literal

if TYPE_CHECKING:
    from .trainer import Trainer
    from .vessel import TrainingVesselBase

_logger = get_module_logger(__name__)


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

    def on_iter_start(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """Called when every iteration (i.e., collect) starts."""

    def on_iter_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """Called upon every end of iteration.
        This is called **after** the bump of ``current_iter``,
        when the previous iteration is considered complete.
        """

    def state_dict(self) -> Any:
        """Get a state dict of the callback for pause and resume."""

    def load_state_dict(self, state_dict: Any) -> None:
        """Resume the callback from a saved state dict."""


class EarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving.

    The earlystopping callback will be triggered each time validation ends.
    It will examine the metrics produced in validation,
    and get the metric with name ``monitor` (``monitor`` is ``reward`` by default),
    to check whether it's no longer increasing / decreasing.
    It takes ``min_delta`` and ``patience`` if applicable.
    If it's found to be not increasing / decreasing any more.
    ``trainer.should_stop`` will be set to true,
    and the training terminates.

    Implementation reference: https://github.com/keras-team/keras/blob/v2.9.0/keras/callbacks.py#L1744-L1893
    """

    def __init__(
        self,
        monitor: str = "reward",
        min_delta: float = 0.0,
        patience: int = 0,
        mode: Literal["min", "max"] = "max",
        baseline: float | None = None,
        restore_best_weights: bool = False,
    ):
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.restore_best_weights = restore_best_weights
        self.best_weights: Any | None = None

        if mode not in ["min", "max"]:
            raise ValueError("Unsupported earlystopping mode: " + mode)

        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def state_dict(self) -> dict:
        return {"wait": self.wait, "best": self.best, "best_weights": self.best_weights, "best_iter": self.best_iter}

    def load_state_dict(self, state_dict: dict) -> None:
        self.wait = state_dict["wait"]
        self.best = state_dict["best"]
        self.best_weights = state_dict["best_weights"]
        self.best_iter = state_dict["best_iter"]

    def on_fit_start(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        # Allow instances to be re-used
        self.wait = 0
        self.best = np.inf if self.monitor_op == np.less else -np.inf
        self.best_weights = None
        self.best_iter = 0

    def on_validate_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        current = self.get_monitor_value(trainer)
        if current is None:
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first iteration if no progress is ever made.
            self.best_weights = copy.deepcopy(vessel.state_dict())

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_iter = trainer.current_iter
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(vessel.state_dict())
            # Only restart wait if we beat both the baseline and our previous best.
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0

        msg = (
            f"#{trainer.current_iter} current reward: {current:.4f}, best reward: {self.best:.4f} in #{self.best_iter}"
        )
        _logger.info(msg)

        # Only check after the first epoch.
        if self.wait >= self.patience and trainer.current_iter > 0:
            trainer.should_stop = True
            _logger.info(f"On iteration %d: early stopping", trainer.current_iter + 1)
            if self.restore_best_weights and self.best_weights is not None:
                _logger.info("Restoring model weights from the end of the best iteration: %d", self.best_iter + 1)
                vessel.load_state_dict(self.best_weights)

    def get_monitor_value(self, trainer: Trainer) -> Any:
        monitor_value = trainer.metrics.get(self.monitor)
        if monitor_value is None:
            _logger.warning(
                "Early stopping conditioned on metric `%s` which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(trainer.metrics.keys())),
            )
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)


class MetricsWriter(Callback):
    """Dump training metrics to file."""

    def __init__(self, dirpath: Path) -> None:
        self.dirpath = dirpath
        self.dirpath.mkdir(exist_ok=True, parents=True)
        self.train_records: List[dict] = []
        self.valid_records: List[dict] = []

    def on_train_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        self.train_records.append({k: v for k, v in trainer.metrics.items() if not k.startswith("val/")})
        pd.DataFrame.from_records(self.train_records).to_csv(self.dirpath / "train_result.csv", index=True)

    def on_validate_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        self.valid_records.append({k: v for k, v in trainer.metrics.items() if k.startswith("val/")})
        pd.DataFrame.from_records(self.valid_records).to_csv(self.dirpath / "validation_result.csv", index=True)


class Checkpoint(Callback):
    """Save checkpoints periodically for persistence and recovery.

    Reference: https://github.com/PyTorchLightning/pytorch-lightning/blob/bfa8b7be/pytorch_lightning/callbacks/model_checkpoint.py

    Parameters
    ----------
    dirpath
        Directory to save the checkpoint file.
    filename
        Checkpoint filename. Can contain named formatting options to be auto-filled.
        For example: ``{iter:03d}-{reward:.2f}.pth``.
        Supported argument names are:

        - iter (int)
        - metrics in ``trainer.metrics``
        - time string, in the format of ``%Y%m%d%H%M%S``
    save_latest
        Save the latest checkpoint in ``latest.pth``.
        If ``link``, ``latest.pth`` will be created as a softlink.
        If ``copy``, ``latest.pth`` will be stored as an individual copy.
        Set to none to disable this.
    every_n_iters
        Checkpoints are saved at the end of every n iterations of training,
        after validation if applicable.
    time_interval
        Maximum time (seconds) before checkpoints save again.
    save_on_fit_end
        Save one last checkpoint at the end to fit.
        Do nothing if a checkpoint is already saved there.
    """

    def __init__(
        self,
        dirpath: Path,
        filename: str = "{iter:03d}.pth",
        save_latest: Literal["link", "copy"] | None = "link",
        every_n_iters: int | None = None,
        time_interval: int | None = None,
        save_on_fit_end: bool = True,
    ):
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.save_latest = save_latest
        self.every_n_iters = every_n_iters
        self.time_interval = time_interval
        self.save_on_fit_end = save_on_fit_end

        self._last_checkpoint_name: str | None = None
        self._last_checkpoint_iter: int | None = None
        self._last_checkpoint_time: float | None = None

    def on_fit_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        if self.save_on_fit_end and (trainer.current_iter != self._last_checkpoint_iter):
            self._save_checkpoint(trainer)

    def on_iter_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        should_save_ckpt = False
        if self.every_n_iters is not None and (trainer.current_iter + 1) % self.every_n_iters == 0:
            should_save_ckpt = True
        if self.time_interval is not None and (
            self._last_checkpoint_time is None or (time.time() - self._last_checkpoint_time) >= self.time_interval
        ):
            should_save_ckpt = True
        if should_save_ckpt:
            self._save_checkpoint(trainer)

    def _save_checkpoint(self, trainer: Trainer) -> None:
        self.dirpath.mkdir(exist_ok=True, parents=True)
        self._last_checkpoint_name = self._new_checkpoint_name(trainer)
        self._last_checkpoint_iter = trainer.current_iter
        self._last_checkpoint_time = time.time()
        torch.save(trainer.state_dict(), self.dirpath / self._last_checkpoint_name)

        latest_pth = self.dirpath / "latest.pth"

        # Remove first before saving
        if self.save_latest and (latest_pth.exists() or os.path.islink(latest_pth)):
            latest_pth.unlink()

        if self.save_latest == "link":
            latest_pth.symlink_to(self.dirpath / self._last_checkpoint_name)
        elif self.save_latest == "copy":
            shutil.copyfile(self.dirpath / self._last_checkpoint_name, latest_pth)

    def _new_checkpoint_name(self, trainer: Trainer) -> str:
        return self.filename.format(
            iter=trainer.current_iter, time=datetime.now().strftime("%Y%m%d%H%M%S"), **trainer.metrics
        )

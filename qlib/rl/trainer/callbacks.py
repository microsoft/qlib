# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Callbacks to insert customized recipes during the training.
Mimicks the hooks of Keras / PyTorch-Lightning, but tailored for the context of RL.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .trainer import Trainer
    from .vessel import TrainingVesselBase


class Callback:
    """Base class of all callbacks."""

    def on_train_start(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """Called when each collect for training begins."""

    def on_train_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """Called when the training ends.
        To access all outputs produced during training, cache the data in either trainer and vessel,
        and post-process them in this hook.
        """

    def on_validate_start(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """Called when each collect for validation begins."""

    def on_validate_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """Called when the validation ends."""

    def on_test_start(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """Called when each collect for testing begins."""

    def on_test_end(self, trainer: Trainer, vessel: TrainingVesselBase) -> None:
        """Called when the testing ends."""


class EarlyStopping(Callback):

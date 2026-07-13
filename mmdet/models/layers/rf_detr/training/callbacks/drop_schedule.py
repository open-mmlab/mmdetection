# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Drop-path / dropout schedule callback for RF-DETR Lightning training."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pytorch_lightning import Callback, LightningModule, Trainer

from mmdet.models.layers.rf_detr.training.drop_schedule import drop_scheduler


class DropPathCallback(Callback):
    """Applies per-step drop-path and dropout rate schedules to the model.

    Computes the full schedule array in ``on_train_start`` using :func:`rfdetr.util.drop_scheduler.drop_scheduler`, then
    indexes into it on every training batch to update the model's stochastic-depth and dropout rates.

    Args:
        drop_path: Peak drop-path rate.  ``0.0`` disables the schedule.
        dropout: Peak dropout rate.  ``0.0`` disables the schedule.
        cutoff_epoch: Epoch boundary for *early* / *late* modes.
        mode: Schedule mode forwarded to ``drop_scheduler``.
        schedule: Schedule shape forwarded to ``drop_scheduler``.
        vit_encoder_num_layers: Passed to ``model.update_drop_path`` so the
            model can distribute rates across ViT encoder layers.
    """

    def __init__(
        self,
        drop_path: float = 0.0,
        dropout: float = 0.0,
        cutoff_epoch: int = 0,
        mode: Literal["standard", "early", "late"] = "standard",
        schedule: Literal["constant", "linear"] = "constant",
        vit_encoder_num_layers: int = 12,
    ) -> None:
        super().__init__()
        self._drop_path = drop_path
        self._dropout = dropout
        self._cutoff_epoch = cutoff_epoch
        self._mode = mode
        self._schedule = schedule
        self._vit_encoder_num_layers = vit_encoder_num_layers

        self._dp_schedule: np.ndarray | None = None
        self._do_schedule: np.ndarray | None = None

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Build per-step rate arrays from trainer metadata.

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The ``RFDETRModelModule`` being trained.
        """
        epochs: int = pl_module.train_config.epochs
        total_steps = int(trainer.estimated_stepping_batches)
        steps_per_epoch = max(1, total_steps // epochs)

        if self._drop_path > 0:
            self._dp_schedule = drop_scheduler(
                self._drop_path,
                epochs,
                steps_per_epoch,
                self._cutoff_epoch,
                self._mode,
                self._schedule,
            )

        if self._dropout > 0:
            self._do_schedule = drop_scheduler(
                self._dropout,
                epochs,
                steps_per_epoch,
                self._cutoff_epoch,
                self._mode,
                self._schedule,
            )

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Apply the scheduled rates for the current global step.

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The ``RFDETRModelModule`` being trained.
            batch: The current training batch (unused).
            batch_idx: Index of the current batch within the epoch (unused).
        """
        step: int = trainer.global_step

        if self._dp_schedule is not None and step < len(self._dp_schedule):
            pl_module.model.update_drop_path(self._dp_schedule[step], self._vit_encoder_num_layers)

        if self._do_schedule is not None and step < len(self._do_schedule):
            pl_module.model.update_dropout(self._do_schedule[step])

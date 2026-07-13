# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""RF-DETR training package (PyTorch Lightning).

Provides the Lightning module, data module, callbacks, and CLI for training and evaluation.

Exports:
    RFDETRModelModule: LightningModule wrapping the RF-DETR model and training loop.
    RFDETRDataModule: LightningDataModule wrapping dataset construction and loaders.
    build_trainer: Factory that assembles a PTL Trainer from RF-DETR configs.
"""

from pytorch_lightning import seed_everything

from mmdet.models.layers.rf_detr.training.callbacks import (
    BestModelCallback,
    COCOEvalCallback,
    DropPathCallback,
    RFDETREarlyStopping,
    RFDETREMACallback,
)
from mmdet.models.layers.rf_detr.training.checkpoint import convert_legacy_checkpoint
from mmdet.models.layers.rf_detr.training.cli import RFDETRCli
from mmdet.models.layers.rf_detr.training.module_data import RFDETRDataModule
from mmdet.models.layers.rf_detr.training.module_model import RFDETRModelModule
from mmdet.models.layers.rf_detr.training.trainer import build_trainer
from mmdet.models.layers.rf_detr.utilities.logger import get_logger

_logger = get_logger()

__all__ = [
    "BestModelCallback",
    "COCOEvalCallback",
    "DropPathCallback",
    "RFDETRCli",
    "RFDETRDataModule",
    "RFDETREMACallback",
    "RFDETREarlyStopping",
    "RFDETRModelModule",
    "build_trainer",
    "convert_legacy_checkpoint",
    "seed_everything",
]

# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Backward-compatibility shim — rfdetr.util.drop_scheduler is deprecated; use rfdetr.training.drop_schedule."""

from mmdet.models.layers.rf_detr.utilities.decorators import _warn_deprecated_module

_warn_deprecated_module(
    "rfdetr.util.drop_scheduler", "rfdetr.training.drop_schedule", deprecated_in="1.6.0", remove_in="1.9.0"
)

from mmdet.models.layers.rf_detr.training.drop_schedule import drop_scheduler  # noqa: F401, E402

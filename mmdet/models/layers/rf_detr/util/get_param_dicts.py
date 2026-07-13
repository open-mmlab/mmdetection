# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Backward-compatibility shim — rfdetr.util.get_param_dicts is deprecated; use rfdetr.training.param_groups."""

from mmdet.models.layers.rf_detr.utilities.decorators import _warn_deprecated_module

_warn_deprecated_module(
    "rfdetr.util.get_param_dicts", "rfdetr.training.param_groups", deprecated_in="1.6.0", remove_in="1.9.0"
)

from mmdet.models.layers.rf_detr.training.param_groups import (  # noqa: F401, E402
    get_param_dict,
    get_vit_lr_decay_rate,
    get_vit_weight_decay_rate,
)

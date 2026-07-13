# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Deprecated: use ``rfdetr.utilities.box_ops`` instead."""

from mmdet.models.layers.rf_detr.utilities.decorators import _warn_deprecated_module

_warn_deprecated_module("rfdetr.util.box_ops", "rfdetr.utilities.box_ops", deprecated_in="1.6.0", remove_in="1.9.0")

from mmdet.models.layers.rf_detr.utilities.box_ops import (  # noqa: F401, E402
    batch_dice_loss,
    batch_dice_loss_jit,
    batch_sigmoid_ce_loss,
    batch_sigmoid_ce_loss_jit,
    box_cxcywh_to_xyxy,
    box_iou,
    box_xyxy_to_cxcywh,
    generalized_box_iou,
    masks_to_boxes,
)

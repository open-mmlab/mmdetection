# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Framework-agnostic evaluation utilities for RF-DETR."""

from mmdet.models.layers.rf_detr.evaluation.keypoint_oks import (
    DEFAULT_KEYPOINT_MAX_DETS,
    MetricKeypointOKS,
    OKSKey,
)

__all__ = [
    "DEFAULT_KEYPOINT_MAX_DETS",
    "OKSKey",
    "MetricKeypointOKS",
]

# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Utility functions and helpers."""

from mmdet.models.layers.rf_detr.utilities import box_ops
from mmdet.models.layers.rf_detr.utilities.distributed import (
    all_gather,
    get_rank,
    get_world_size,
    is_dist_avail_and_initialized,
    is_main_process,
    reduce_dict,
    save_on_master,
)
from mmdet.models.layers.rf_detr.utilities.keypoints import (
    precision_cholesky_to_pixel_covariance,
    schemas_semantically_equal,
)
from mmdet.models.layers.rf_detr.utilities.logger import get_logger
from mmdet.models.layers.rf_detr.utilities.package import get_sha, get_version
from mmdet.models.layers.rf_detr.utilities.reproducibility import seed_all
from mmdet.models.layers.rf_detr.utilities.state_dict import clean_state_dict, strip_checkpoint
from mmdet.models.layers.rf_detr.utilities.tensors import (
    NestedTensor,
    collate_fn,
    make_collate_fn,
    nested_tensor_from_tensor_list,
)

__all__ = [
    # distributed
    "all_gather",
    "get_rank",
    "get_world_size",
    "is_dist_avail_and_initialized",
    "is_main_process",
    "reduce_dict",
    "save_on_master",
    # tensors
    "NestedTensor",
    "collate_fn",
    "make_collate_fn",
    "nested_tensor_from_tensor_list",
    # box_ops (submodule)
    "box_ops",
    # logger
    "get_logger",
    # package
    "get_sha",
    "get_version",
    # keypoints
    "schemas_semantically_equal",
    "precision_cholesky_to_pixel_covariance",
    # reproducibility
    "seed_all",
    # state_dict
    "clean_state_dict",
    "strip_checkpoint",
]

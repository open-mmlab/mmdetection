# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
"""ms_deform_attn_func."""

from __future__ import annotations

import torch
from torch import Tensor

from mmdet.models.layers.rf_detr.utilities.tensors import _bilinear_grid_sample


def ms_deform_attn_core_pytorch(
    value: Tensor,
    value_spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    value_spatial_shapes_hw: list[tuple[int, int]] | None = None,
) -> Tensor:
    """For debug and test only, need to use cuda version instead."""
    # batch_size, n_heads, head_dim, spatial_size
    batch_size, n_heads, head_dim, _ = value.shape
    _, len_query, n_heads, num_levels, num_points, _ = sampling_locations.shape
    # Use Python int pairs when available (required for torch.export compatibility,
    # since iterating over a tensor and using scalar elements as split/view sizes
    # fails during FakeTensor tracing).
    shapes = value_spatial_shapes_hw if value_spatial_shapes_hw is not None else value_spatial_shapes
    split_sizes = [int(height) * int(width) for height, width in shapes]
    value_list = value.split(split_sizes, dim=3)  # type: ignore[no-untyped-call]
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_index, (height, width) in enumerate(shapes):
        # batch_size, n_heads, head_dim, height, width
        value_l_ = value_list[level_index].view(batch_size * n_heads, head_dim, height, width)
        # batch_size, len_query, n_heads, num_points, 2
        # -> batch_size, n_heads, len_query, num_points, 2
        # -> batch_size*n_heads, len_query, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level_index].transpose(1, 2).flatten(0, 1)
        # batch_size*n_heads, head_dim, len_query, num_points
        sampling_value_l_ = _bilinear_grid_sample(value_l_, sampling_grid_l_, padding_mode="zeros", align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (batch_size, len_query, n_heads, num_levels * num_points)
    # -> (batch_size, n_heads, len_query, num_levels, num_points)
    # -> (batch_size*n_heads, 1, len_query, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * n_heads, 1, len_query, num_levels * num_points
    )
    # batch_size*n_heads, head_dim, len_query, num_levels*num_points
    sampling_values = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    output = (sampling_values * attention_weights).sum(-1).view(batch_size, n_heads * head_dim, len_query)
    return output.transpose(1, 2).contiguous()

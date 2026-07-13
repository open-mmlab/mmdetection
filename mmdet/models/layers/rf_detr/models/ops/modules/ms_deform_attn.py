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
"""Multi-Scale Deformable Attention Module."""

from __future__ import annotations

import math
import warnings
from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.nn.init import constant_, xavier_uniform_

from mmdet.models.layers.rf_detr.models.ops.functions import ms_deform_attn_core_pytorch


def _is_power_of_2(n: int) -> bool:
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(f"invalid input for _is_power_of_2: {n} (type: {type(n)})")
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention Module."""

    def __init__(self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4) -> None:
        """Multi-Scale Deformable Attention Module :param d_model      hidden dimension :param n_levels     number of
        feature levels :param n_heads      number of attention heads :param n_points     number of sampling points per
        attention head per feature level."""
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the"
                " dimension of each attention head a power of 2"
                " which is more efficient in our CUDA implementation."
            )

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

        self._export = False

    def export(self) -> None:
        """Export mode."""
        self._export = True

    def _reset_parameters(self) -> None:
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        input_flatten: Tensor,
        input_spatial_shapes: Tensor,
        input_level_start_index: Tensor,
        input_padding_mask: Tensor | None = None,
        input_spatial_shapes_hw: list[tuple[int, int]] | None = None,
    ) -> Tensor:
        """Forward pass of MSDeformAttn.

        Args:
            query: (N, Length_{query}, C)
            reference_points: (N, Length_{query}, n_levels, 2) with range in [0, 1],
                top-left (0,0), bottom-right (1, 1), including padding area; or (N, Length_{query}, n_levels, 4) adding
                additional (w, h) to form reference boxes.
            input_flatten: (N, sum_{l=0}^{L-1} H_l * W_l, C)
            input_spatial_shapes: (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            input_level_start_index: (n_levels,), [0, H_0*W_0, H_0*W_0+H_1*W_1, ...,
                H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
            input_padding_mask: (N, sum_{l=0}^{L-1} H_l * W_l), True for padding elements,
                False for non-padding elements.
            input_spatial_shapes_hw: List of (H, W) int pairs, same ordering as
                input_spatial_shapes. When provided, these Python ints are used for tensor split/view operations inside
                ms_deform_attn_core_pytorch so that the function is compatible with torch.export.export (FakeTensor
                tracing cannot extract concrete values from a tensor).

        Returns:
            Output tensor of shape (N, Length_{query}, C).
        """
        batch_size, len_query, _ = query.shape
        batch_size, len_input, _ = input_flatten.shape
        expected_len_in = (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum()
        error_msg = "input_spatial_shapes must match the flattened input length"
        if self._export:
            torch._assert(expected_len_in == len_input, error_msg)  # type: ignore[no-untyped-call]
        else:
            assert expected_len_in == len_input, error_msg

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        sampling_offsets = self.sampling_offsets(query).view(
            batch_size, len_query, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            batch_size, len_query, self.n_heads, self.n_levels * self.n_points
        )

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead."
            )
        attention_weights = F.softmax(attention_weights, -1)

        value = (
            value.transpose(1, 2).contiguous().view(batch_size, self.n_heads, self.d_model // self.n_heads, len_input)
        )
        output = ms_deform_attn_core_pytorch(
            value,
            input_spatial_shapes,
            sampling_locations,
            attention_weights,
            value_spatial_shapes_hw=input_spatial_shapes_hw,
        )
        return cast(Tensor, self.output_proj(output))

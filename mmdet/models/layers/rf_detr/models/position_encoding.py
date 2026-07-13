# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""Various positional encodings for the transformer."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from mmdet.models.layers.rf_detr.utilities.tensors import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """This is a more standard version of the position embedding, very similar to the one used by the Attention is all
    you need paper, generalized to work on images."""

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self._export = False

    def export(self) -> None:
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export  # type: ignore[method-assign,assignment]

    def forward(self, tensor_list: NestedTensor, align_dim_orders: bool = True) -> Tensor:
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        if align_dim_orders:
            pos = torch.cat((pos_y, pos_x), dim=3).permute(1, 2, 0, 3)
            # return: (H, W, bs, C)
        else:
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
            # return: (bs, C, H, W)
        return pos

    def forward_export(self, mask: Tensor, align_dim_orders: bool = True) -> Tensor:
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        if align_dim_orders:
            pos = torch.cat((pos_y, pos_x), dim=3).permute(1, 2, 0, 3)
            # return: (H, W, bs, C)
        else:
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
            # return: (bs, C, H, W)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, num_pos_feats: int = 256) -> None:
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()
        self._export = False

    def export(self) -> None:
        raise NotImplementedError

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor) -> Tensor:
        x = tensor_list.tensors
        h, w = x.shape[:2]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .unsqueeze(2)
            .repeat(1, 1, x.shape[2], 1)
        )
        # return: (H, W, bs, C)
        return pos


def build_position_encoding(hidden_dim: int, position_embedding: str) -> nn.Module:
    """Build a positional encoding module.

    Args:
        hidden_dim: Transformer hidden dimension. Half of this value is used as the number
            of positional feature dimensions for the sine encoding.
        position_embedding: Encoding variant to construct.  Supported values:
            ``"sine"`` / ``"v2"`` — standard sine/cosine positional encoding (recommended).
            The aliases ``"learned"`` / ``"v3"`` are **not supported** and raise
            :exc:`ValueError`; their implementation had two bugs (wrong ``forward`` signature
            and wrong shape unpacking) and are rejected early rather than silently producing
            incorrect results.

    Returns:
        Positional encoding module.

    Raises:
        ValueError: If *position_embedding* is not a recognised and supported variant.

    Examples:
        >>> import torch
        >>> from mmdet.models.layers.rf_detr.models.position_encoding import build_position_encoding
        >>> enc = build_position_encoding(256, "sine")
        >>> enc  # doctest: +ELLIPSIS
        PositionEmbeddingSine(...)
    """
    num_steps = hidden_dim // 2
    if position_embedding in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        return PositionEmbeddingSine(num_steps, normalize=True)
    if position_embedding in ("v3", "learned"):
        raise ValueError(
            f"position_embedding={position_embedding!r} is not supported. "
            "The 'learned'/'v3' implementation has two bugs: the forward() signature "
            "is incompatible with Joiner.forward(), and the shape unpacking uses "
            "x.shape[:2] (batch, channels) instead of x.shape[-2:] (height, width). "
            "Use 'sine' or 'v2' instead."
        )
    raise ValueError(f"position_embedding={position_embedding!r} is not supported. Supported values: 'sine', 'v2'.")

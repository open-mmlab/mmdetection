# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from ViTDet (https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""Projector."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class LayerNorm(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs point-wise mean and variance normalization over
    the channel dimension for inputs that have shape (batch_size, channels, height, width).

    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: Tensor) -> Tensor:
        """
        LayerNorm forward
        TODO: this is a hack to avoid overflow when using fp16
        """
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


def get_norm(norm: str | Callable[[int], nn.Module] | None, out_channels: int) -> nn.Module | None:
    """
    Args:
        norm: Either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns the normalization layer as a nn.Module.

    Returns:
        The normalization layer.
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "LN": lambda channels: LayerNorm(channels),
        }[norm]
    return norm(out_channels)


def get_activation(name: str | None, inplace: bool = False) -> nn.Module:
    """Get activation."""
    module: nn.Module
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name in ["LeakyReLU", "leakyrelu", "lrelu"]:
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name is None:
        module = nn.Identity()
    else:
        raise AttributeError(f"Unsupported act type: {name}")
    return module


class ConvX(nn.Module):
    """Conv-bn module."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel: int | tuple[int, int] = 3,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        act: str = "relu",
        layer_norm: bool = False,
        rms_norm: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(kernel, tuple):
            kernel = (kernel, kernel)
        padding = (kernel[0] // 2, kernel[1] // 2)
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn: nn.Module
        if rms_norm:
            self.bn = nn.RMSNorm(out_planes)
        else:
            self.bn = cast(nn.Module, get_norm("LN", out_planes)) if layer_norm else nn.BatchNorm2d(out_planes)
        self.act = get_activation(act, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """forward."""
        out = self.act(self.bn(self.conv(x.contiguous())))
        return cast(Tensor, out)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
        act: str = "silu",
        layer_norm: bool = False,
        rms_norm: bool = False,
    ) -> None:
        """ch_in, ch_out, shortcut, groups, kernels, expand."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvX(c1, c_, k[0], 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.cv2 = ConvX(c_, c2, k[1], 1, groups=g, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.add = shortcut and c1 == c2

    def forward(self, x: Tensor) -> Tensor:
        """'forward()' applies the YOLOv5 FPN to input data."""
        return cast(Tensor, x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x)))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        act: str = "silu",
        layer_norm: bool = False,
        rms_norm: bool = False,
    ) -> None:
        """ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvX(c1, 2 * self.c, 1, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.cv2 = ConvX(
            (2 + n) * self.c, c2, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm
        )  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
            for _ in range(n)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return cast(Tensor, self.cv2(torch.cat(y, 1)))


class MultiScaleProjector(nn.Module):
    """This module implements MultiScaleProjector in :paper:`lwdetr`.

    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        scale_factors: Sequence[float],
        num_blocks: int = 3,
        layer_norm: bool = False,
        rms_norm: bool = False,
        survival_prob: float = 1.0,
        force_drop_last_n_features: int = 0,
    ) -> None:
        """
        Args:
            in_channels: Channels in each input feature map level.
            out_channels: Number of channels in the output feature maps.
            scale_factors: List of scaling factors to upsample or downsample
                the input features for creating pyramid features.
        """
        super().__init__()

        self.scale_factors = scale_factors
        self.survival_prob = survival_prob
        self.force_drop_last_n_features = force_drop_last_n_features

        stages_sampling: list[nn.ModuleList] = []
        stages: list[nn.Sequential] = []
        # use_bias = norm == ""
        self.use_extra_pool = False
        for scale in scale_factors:
            scale_stage_layers: list[nn.Module] = []
            for in_dim in in_channels:
                layers: list[nn.Module] = []

                # if in_dim > 512:
                #     layers.append(ConvX(in_dim, in_dim // 2, kernel=1))
                #     in_dim = in_dim // 2

                if scale == 4.0:
                    layers.extend(
                        [
                            nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                            cast(nn.Module, get_norm("LN", in_dim // 2)),
                            nn.GELU(),
                            nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
                        ]
                    )
                    # in_dim // 4
                elif scale == 2.0:
                    # a hack to reduce the FLOPs and Params when the dimension of output feature is too large
                    # if in_dim > 512:
                    #     layers = [
                    #         ConvX(in_dim, in_dim // 2, kernel=1),
                    #         nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
                    #     ]
                    #     out_dim = in_dim // 4
                    # else:
                    layers.extend(
                        [
                            nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                        ]
                    )
                    # in_dim // 2
                elif scale == 1.0:
                    pass
                elif scale == 0.5:
                    layers.extend(
                        [
                            ConvX(in_dim, in_dim, 3, 2, layer_norm=layer_norm),
                        ]
                    )
                elif scale == 0.25:
                    self.use_extra_pool = True
                    continue
                else:
                    raise NotImplementedError(f"Unsupported scale_factor:{scale}")
                scale_stage_layers.append(nn.Sequential(*layers))
            stages_sampling.append(nn.ModuleList(scale_stage_layers))

            in_dim = int(sum(in_channel // max(1, scale) for in_channel in in_channels))
            stage_layers: list[nn.Module] = [
                C2f(in_dim, out_channels, num_blocks, layer_norm=layer_norm),
                cast(nn.Module, get_norm("LN", out_channels)),
            ]
            stages.append(nn.Sequential(*stage_layers))

        self.stages_sampling = nn.ModuleList(stages_sampling)
        self.stages = nn.ModuleList(stages)

    def forward(self, x: list[Tensor]) -> list[Tensor]:
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        num_features = len(x)
        if self.survival_prob < 1.0 and self.training:
            x = list(x)  # copy before mutating so the caller's list is untouched
            final_drop_prob = 1 - self.survival_prob
            # torch RNG (not numpy) so the draw honours per-rank seeding under DDP.
            drop_p = torch.rand(()).item()
            for i in range(1, num_features):
                critical_drop_prob = i * (final_drop_prob / (num_features - 1))
                if drop_p < critical_drop_prob:
                    x[i] = torch.zeros_like(x[i])
        elif self.force_drop_last_n_features > 0:
            for i in range(self.force_drop_last_n_features):
                # don't do it inplace to ensure the compiler can optimize out the backbone layers
                x[-(i + 1)] = torch.zeros_like(x[-(i + 1)])

        results = []
        # x list of len(out_features_indexes)
        for i, stage in enumerate(self.stages):
            feat_fuse_list = []
            for j, stage_sampling in enumerate(cast(nn.ModuleList, self.stages_sampling[i])):
                feat_fuse_list.append(stage_sampling(x[j]))
            feat_fuse = torch.cat(feat_fuse_list, dim=1) if len(feat_fuse_list) > 1 else feat_fuse_list[0]
            results.append(stage(feat_fuse))
        if self.use_extra_pool:
            results.append(F.max_pool2d(results[-1], kernel_size=1, stride=2, padding=0))
        return results

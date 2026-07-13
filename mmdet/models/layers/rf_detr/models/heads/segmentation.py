# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.nn.grad import conv2d_input, conv2d_weight

from mmdet.models.layers.rf_detr.utilities.tensors import _bilinear_grid_sample


class _DepthwiseConvWithoutCuDNN(torch.autograd.Function):
    """Depthwise conv2d with cuDNN disabled in both forward and backward.

    ``torch.backends.cudnn.flags(enabled=False)`` as a context manager only covers operations executed within its scope.
    ``nn.Conv2d`` records the forward op in the autograd graph; the corresponding backward kernels run later,
    **outside** that scope, with cuDNN re-enabled.  On some CUDA stacks (T4 / P100 on Kaggle / Colab) cuDNN fails engine
    selection for depthwise conv backward, raising::

        RuntimeError: GET was unable to find an engine to execute this computation

    This ``Function`` disables cuDNN in ``backward`` as well, fixing the crash.

    See: https://github.com/roboflow/rf-detr/issues/731
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        stride: tuple[int, ...],
        padding: tuple[int, ...],
        dilation: tuple[int, ...],
        groups: int,
    ) -> Tensor:
        """Run depthwise conv2d forward with cuDNN disabled.

        Args:
            ctx: Autograd context.
            x: Input feature map ``(N, C, H, W)``.
            weight: Convolution weight tensor.
            bias: Optional convolution bias tensor.
            stride: Convolution stride.
            padding: Convolution padding.
            dilation: Convolution dilation.
            groups: Number of groups (equals ``C`` for depthwise).

        Returns:
            Output feature map ``(N, C, H, W)``.
        """
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None  # type: ignore[attr-defined]
        ctx.stride = stride  # type: ignore[attr-defined]
        ctx.padding = padding  # type: ignore[attr-defined]
        ctx.dilation = dilation  # type: ignore[attr-defined]
        ctx.groups = groups  # type: ignore[attr-defined]
        # Note: torch.backends.cudnn.flags() is process-global state, not op-local.
        # Safe under DDP (separate processes per rank), but concurrent backward passes
        # in the same process (DataParallel, user threads) could briefly observe the
        # wrong cuDNN setting.  For DDP-only training this is not a concern.
        with torch.backends.cudnn.flags(enabled=False):
            return F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Tensor,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, None, None, None, None]:
        """Compute gradients with cuDNN disabled.

        Args:
            ctx: Autograd context with saved tensors and conv parameters.
            grad_output: Upstream gradient ``(N, C, H, W)``.

        Returns:
            Gradients for each ``forward`` input.  Inputs that do not require gradients (``ctx.needs_input_grad[i]`` is
            ``False``) get ``None``. Non-tensor inputs always get ``None``.

        Note:
            Under AMP (``"bf16-mixed"`` or ``"16-mixed"``), ``grad_output`` may arrive in a reduced dtype while the
            saved ``weight`` stays ``fp32``.  Both tensors are upcast to ``weight.dtype`` before calling
            ``conv2d_input`` / ``conv2d_weight``.  ``grad_input`` is kept in ``weight.dtype`` (fp32) so that upstream
            gradient accumulation into fp32 leaf parameters stays in fp32 — matching standard ``F.conv2d`` backward
            behaviour.  Casting back to the activation dtype (``x.dtype``) would propagate a reduced-precision gradient
            to fp32 backbone parameters, causing a ``params, grads, exp_avgs, and exp_avg_sqs must have same dtype``
            crash in fused AdamW (see issue #959).
        """
        x, weight = ctx.saved_tensors  # type: ignore[attr-defined]

        needs_x_grad = ctx.needs_input_grad[0]  # type: ignore[attr-defined]
        needs_w_grad = ctx.needs_input_grad[1]  # type: ignore[attr-defined]
        needs_b_grad = ctx.has_bias and ctx.needs_input_grad[2]  # type: ignore[attr-defined]

        grad_input = None
        grad_weight = None
        grad_bias = None

        if needs_x_grad or needs_w_grad:
            # Under AMP, grad_output may arrive in a reduced dtype (fp16/bf16) while
            # weight stays fp32.  conv2d_input/conv2d_weight require matching dtypes,
            # so upcast to weight.dtype (fp32).  grad_input is kept in weight.dtype —
            # casting back to x.dtype would inject a bf16 gradient into fp32 params.
            grad_output_cast = grad_output.to(dtype=weight.dtype)
            # Same process-global caveat as forward: safe under DDP, not under DataParallel.
            with torch.backends.cudnn.flags(enabled=False):
                if needs_x_grad:
                    grad_input = conv2d_input(  # type: ignore[no-untyped-call]
                        x.shape,
                        weight,
                        grad_output_cast,
                        stride=ctx.stride,  # type: ignore[attr-defined]
                        padding=ctx.padding,  # type: ignore[attr-defined]
                        dilation=ctx.dilation,  # type: ignore[attr-defined]
                        groups=ctx.groups,  # type: ignore[attr-defined]
                    )  # kept in weight.dtype (fp32) — do NOT cast back to x.dtype
                if needs_w_grad:
                    grad_weight = conv2d_weight(  # type: ignore[no-untyped-call]
                        x.to(dtype=weight.dtype),
                        weight.shape,
                        grad_output_cast,
                        stride=ctx.stride,  # type: ignore[attr-defined]
                        padding=ctx.padding,  # type: ignore[attr-defined]
                        dilation=ctx.dilation,  # type: ignore[attr-defined]
                        groups=ctx.groups,  # type: ignore[attr-defined]
                    )

        if needs_b_grad:
            grad_bias = grad_output.to(dtype=weight.dtype).sum(dim=(0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None, None, None


class DepthwiseConvBlock(nn.Module):
    r"""Simplified ConvNeXt block without the MLP subnet."""

    def __init__(self, dim: int, layer_scale_init_value: float = 0) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def _depthwise_conv(self, x: Tensor) -> Tensor:
        # Custom autograd Function so cuDNN is disabled in both forward AND
        # backward.  A plain context-manager only covers forward; the backward
        # for nn.Conv2d runs outside that scope and re-enables cuDNN,
        # triggering RuntimeError on T4/P100 GPUs (issue #731).
        return cast(
            Tensor,
            _DepthwiseConvWithoutCuDNN.apply(  # type: ignore[no-untyped-call]
                x,
                self.dwconv.weight,
                self.dwconv.bias,
                self.dwconv.stride,
                self.dwconv.padding,
                self.dwconv.dilation,
                self.dwconv.groups,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        input = x
        x = self._depthwise_conv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        return x + input


class MLPBlock(nn.Module):
    def __init__(self, dim: int, layer_scale_init_value: float = 0) -> None:
        super().__init__()
        self.norm_in = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            ]
        )
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        input = x
        x = self.norm_in(x)
        for layer in self.layers:
            x = layer(x)
        if self.gamma is not None:
            x = self.gamma * x
        return x + input


class SegmentationHead(nn.Module):
    def __init__(self, in_dim: int, num_blocks: int, bottleneck_ratio: int = 1, downsample_ratio: int = 4) -> None:
        super().__init__()

        self.downsample_ratio = downsample_ratio
        self.interaction_dim = in_dim // bottleneck_ratio if bottleneck_ratio is not None else in_dim
        self.blocks = nn.ModuleList([DepthwiseConvBlock(in_dim) for _ in range(num_blocks)])
        self.spatial_features_proj = (
            nn.Identity() if bottleneck_ratio is None else nn.Conv2d(in_dim, self.interaction_dim, kernel_size=1)
        )

        self.query_features_block = MLPBlock(in_dim)
        self.query_features_proj = (
            nn.Identity() if bottleneck_ratio is None else nn.Linear(in_dim, self.interaction_dim)
        )

        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)

        self._export = False

    def export(self) -> None:
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export  # type: ignore[method-assign]
        for name, m in self.named_modules():
            if hasattr(m, "export") and callable(m.export) and hasattr(m, "_export") and not m._export:
                m.export()

    def forward(
        self,
        spatial_features: Tensor,
        query_features: list[Tensor],
        image_size: tuple[int, int],
        skip_blocks: bool = False,
    ) -> list[Tensor]:
        # spatial features: (B, C, H, W)
        # query features: [(B, N, C)] for each decoder layer
        # output: (B, N, H*r, W*r)
        target_size = (image_size[0] // self.downsample_ratio, image_size[1] // self.downsample_ratio)
        spatial_features = F.interpolate(spatial_features, size=target_size, mode="bilinear", align_corners=False)

        mask_logits = []
        if not skip_blocks:
            for block, qf in zip(self.blocks, query_features):
                spatial_features = block(spatial_features)
                spatial_features_proj = self.spatial_features_proj(spatial_features)
                qf = self.query_features_proj(self.query_features_block(qf))
                mask_logits.append(torch.einsum("bchw,bnc->bnhw", spatial_features_proj, qf) + self.bias)
        else:
            assert len(query_features) == 1, "skip_blocks is only supported for length 1 query features"
            qf = self.query_features_proj(self.query_features_block(query_features[0]))
            mask_logits.append(torch.einsum("bchw,bnc->bnhw", spatial_features, qf) + self.bias)

        return mask_logits

    def sparse_forward(
        self,
        spatial_features: Tensor,
        query_features: list[Tensor],
        image_size: tuple[int, int],
        skip_blocks: bool = False,
    ) -> list[dict[str, Tensor]]:
        # spatial features: (B, C, H, W)
        # query features: [(B, N, C)] for each decoder layer
        # output: dict containing the intermediate results
        target_size = (image_size[0] // self.downsample_ratio, image_size[1] // self.downsample_ratio)
        spatial_features = F.interpolate(spatial_features, size=target_size, mode="bilinear", align_corners=False)

        # num_points = max(spatial_features.shape[-2], spatial_features.shape[-2] * spatial_features.shape[-1] // 16)

        output_dicts: list[dict[str, Tensor]] = []

        if not skip_blocks:
            for block, qf in zip(self.blocks, query_features):
                spatial_features = block(spatial_features)
                spatial_features_proj = self.spatial_features_proj(spatial_features)
                qf = self.query_features_proj(self.query_features_block(qf))

                output_dicts.append(
                    {
                        "spatial_features": spatial_features_proj,
                        "query_features": qf,
                        "bias": self.bias,
                    }
                )
        else:
            assert len(query_features) == 1, "skip_blocks is only supported for length 1 query features"

            qf = self.query_features_proj(self.query_features_block(query_features[0]))

            output_dicts.append(
                {
                    "spatial_features": spatial_features,
                    "query_features": qf,
                    "bias": self.bias,
                }
            )

        return output_dicts

    def forward_export(
        self,
        spatial_features: Tensor,
        query_features: list[Tensor],
        image_size: tuple[int, int],
        skip_blocks: bool = False,
    ) -> list[Tensor]:
        assert len(query_features) == 1, "at export time, segmentation head expects exactly one query feature"

        target_size = (image_size[0] // self.downsample_ratio, image_size[1] // self.downsample_ratio)
        spatial_features = F.interpolate(spatial_features, size=target_size, mode="bilinear", align_corners=False)

        if not skip_blocks:
            for block in self.blocks:
                spatial_features = block(spatial_features)

        spatial_features_proj = self.spatial_features_proj(spatial_features)

        qf = self.query_features_proj(self.query_features_block(query_features[0]))
        return [torch.einsum("bchw,bnc->bnhw", spatial_features_proj, qf) + self.bias]


def point_sample(input: Tensor, point_coords: Tensor, **kwargs: Any) -> Tensor:
    """A wrapper around :func:`~rfdetr.utilities.tensors._bilinear_grid_sample` to support 3D point_coords tensors.
    Unlike :func:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside [0, 1] x [0, 1] square.

    Args:
        input: A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords: A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
            [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interpolation from `input` the same way as :func:`~rfdetr.utilities.tensors._bilinear_grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)

    # Normalize coordinates from [0, 1] to [-1, 1] as expected by grid_sample.
    grid = 2.0 * point_coords - 1.0

    # Extract common grid_sample arguments, with bilinear as the default mode to
    # preserve existing behavior when mode is not provided.
    mode = kwargs.pop("mode", "bilinear")
    align_corners = kwargs.pop("align_corners", False)
    padding_mode = kwargs.pop("padding_mode", "border")

    if mode == "bilinear":
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword argument(s) for bilinear mode: {unexpected}")
        # For bilinear mode, use the optimized sampler when the padding_mode
        # is supported by the manual/MPS path. For other padding modes,
        # delegate to F.grid_sample to keep behavior consistent across devices.
        if padding_mode not in ("zeros", "border"):
            output = F.grid_sample(
                input,
                grid,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
            )
        else:
            output = _bilinear_grid_sample(
                input,
                grid,
                padding_mode=padding_mode,
                align_corners=align_corners,
            )
    else:
        # Delegate to torch.nn.functional.grid_sample for other modes (e.g. "nearest"),
        # forwarding any remaining supported kwargs.
        output = F.grid_sample(
            input,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            **kwargs,
        )

    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertain_point_coords_with_randomness(
    coarse_logits: Tensor,
    uncertainty_func: Callable[[Tensor], Tensor],
    num_points: int,
    oversample_ratio: int = 3,
    importance_sample_ratio: float = 0.75,
) -> Tensor:
    """Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty.

    The uncertainties are calculated for each point using 'uncertainty_func' function that takes point's logit
    prediction as input. See PointRend paper for details.

    Args:
        coarse_logits: A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of shape (N, 1, P).
        num_points: The number of points P to sample.
        oversample_ratio: Oversampling parameter.
        importance_sample_ratio: Ratio of points that are sampled via importance sampling.

    Returns:
        A tensor of shape (N, P, 2) that contains the coordinates of sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)
    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords


def calculate_uncertainty(logits: Tensor) -> Tensor:
    """We estimate uncertainty as L1 distance between 0.0 and the logit prediction in 'logits' for the foreground class
    in `classes`.

    Args:
        logits: A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is the number of
            foreground classes. The values are logits.

    Returns:
        A tensor of shape (R, 1, ...) that contains uncertainty scores with the most uncertain locations having the
        highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))

# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Keypoint helper primitives for GroupPose-style decoding.

Keypoint predictions are dense ``[N, K, KEYPOINT_PRED_DIM]`` tensors where the trailing channel layout is fixed:

================  ===========  ==========================================================
Slot index        Name         Meaning
================  ===========  ==========================================================
``0``             ``x``        Normalized x coordinate (image-relative).
``1``             ``y``        Normalized y coordinate (image-relative).
``2``             ``findable`` Logit for "annotator could find this keypoint" (``v > 0``).
``3``             ``visible``  Logit for "fully visible" (``v == 2``).
``4`` – ``6``     ``L_*``      Lower-triangular Cholesky parameters ``Lxx``, ``Lxy``, ``Lyy``.
``7``             ``class``    Per-keypoint class-logit contribution aggregated into detection-class logits.
================  ===========  ==========================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812 — conventional PyTorch alias
from torch import Tensor, nn

from mmdet.models.layers.rf_detr.utilities.logger import get_logger

logger = get_logger()

# Number of channels in a keypoint prediction slot — see module docstring for layout.
KEYPOINT_PRED_DIM: int = 8


def modulate(features: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    """Apply AdaLN modulation to a feature tensor.

    Args:
        features: Input feature map to modulate.
        scale: Per-feature scale terms.
        shift: Per-feature shift terms.

    Returns:
        Modulated features with the same shape as ``features``.

    Example:
        Apply modulation to a batch of 2 query tokens, each with 4 channels:

        .. code-block:: python

            features = torch.zeros(2, 4)          # (batch, dim)
            scale    = torch.ones(2, 4) * 0.1     # small positive scale
            shift    = torch.ones(2, 4) * (-0.5)  # constant shift

            out = modulate(features, scale, shift)
            # scale=0.1 → effective multiplier is 1.1; shift=-0.5 applied additively
            # out[0] ≈ tensor([-0.5000, -0.5000, -0.5000, -0.5000])
    """

    return (scale + 1.0) * features + shift


class ConditionalQueryInitializer(nn.Module):
    """Initialize keypoint query tokens with adaptive layer-normalization style modulation."""

    def __init__(self, dim: int, num_queries: int, out_dim: int | None = None) -> None:
        """Create the initializer.

        Args:
            dim: Query conditioning dimensionality.
            num_queries: Number of query tokens to instantiate.
            out_dim: Output embedding size. Defaults to ``dim``.
        """
        super().__init__()
        out_dim = out_dim or dim

        self.queries = nn.Parameter(torch.randn(num_queries, out_dim))
        self.query_norm = nn.LayerNorm(out_dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, out_dim * 3),
        )
        ada_ln_projection = cast("nn.Linear", self.adaLN_modulation[-1])
        nn.init.constant_(ada_ln_projection.weight, 0)
        nn.init.constant_(ada_ln_projection.bias, 0)

        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, query_features: Tensor) -> Tensor:
        """Return modulated query embeddings.

        Args:
            query_features: Tensor of shape ``(B, dim)`` containing conditioning features.

        Returns:
            Tensor of shape ``(B, num_queries, out_dim)`` with initialized keypoint queries.

        Example:
            Initialize 17 keypoint queries conditioned on a batch of 2 detection embeddings:

            .. code-block:: python

                initializer = ConditionalQueryInitializer(dim=256, num_queries=17)
                query_features = torch.randn(2, 256)   # (B=2, dim=256)

                keypoint_queries = initializer(query_features)
                # keypoint_queries.shape == (2, 17, 256)
                # Each of the 17 query slots is independently modulated by the
                # per-detection conditioning vector before being passed to the
                # GroupPose keypoint decoder.
        """

        normed_query_features = self.query_norm(self.queries)
        modulation: Tensor = self.adaLN_modulation(query_features.unsqueeze(-2))
        scale, shift, gate = modulation.chunk(3, dim=-1)
        modulated_query_features = self.out_proj(modulate(normed_query_features, scale, shift)) * gate + self.queries
        return cast("Tensor", modulated_query_features)


def compute_l1_keypoint_loss(
    all_pred_keypoints: Tensor,
    target_keypoints: Tensor,
    target_classes: Tensor,
    target_areas: Tensor,
    num_keypoints_per_class: Sequence[int],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the keypoint loss vector per matched target.

    The tensor layout follows GroupPose-style keypoints where each target class
    defines how many valid keypoints it owns. The returned Gaussian NLL follows
    the r-flow formulation directly:
    ``0.5 * maha2 / area - (log_l11 + log_l22)``. It intentionally omits the
    normal Gaussian constant and does not clamp the precision Cholesky
    parameters, so valid losses may be negative.

    Args:
        all_pred_keypoints: Predicted keypoints with shape ``(N, K_total, >=7)``.
        target_keypoints: Ground truth keypoints with shape ``(N, K_max, 3)``.
        target_classes: Class ids per target with shape ``(N,)``.
        target_areas: Target box areas with shape ``(N,)``.
        num_keypoints_per_class: Number of keypoints per class.

    Returns:
        Tuple of location, findable BCE, visible BCE, and raw Gaussian NLL losses.
        Each tensor has shape ``(n_targets,)``.

    Example:
        Compute losses for 2 matched targets, each with 17 keypoints (COCO layout):

        .. code-block:: python

            n_targets, K, pred_dim = 2, 17, 7
            all_pred_keypoints = torch.randn(n_targets, K, pred_dim)

            # Ground truth: 17 keypoints per target, each (x, y, visibility)
            # visibility: 0=not labeled, 1=labeled but occluded, 2=fully visible
            target_keypoints = torch.rand(n_targets, K, 3)
            target_keypoints[:, :, 2] = 2.0   # mark all keypoints fully visible

            target_classes = torch.zeros(n_targets, dtype=torch.long)  # single class (person)
            target_areas   = torch.tensor([0.05, 0.12])  # normalized box areas

            loc_loss, findable_loss, visible_loss, nll_loss = compute_l1_keypoint_loss(
                all_pred_keypoints,
                target_keypoints,
                target_classes,
                target_areas,
                num_keypoints_per_class=[17],
            )
            # Each output tensor has shape (2,) — one scalar loss per matched target.
            # loc_loss:      area-normalized mean L1 distance for visible keypoints
            # findable_loss: BCE for "annotator could locate this keypoint"
            # visible_loss:  BCE for "keypoint is fully visible (v==2)"
            # nll_loss:      Gaussian NLL incorporating Cholesky uncertainty parameters
    """

    n_targets, total_padded_num_keypoints, pred_dim = all_pred_keypoints.shape
    assert pred_dim >= 7, "Expected all_pred_keypoints last dim >= 7 (x,y,2 logits + 3 chol params)."
    num_classes = len(num_keypoints_per_class)

    if num_classes == 0:
        raise ValueError("num_keypoints_per_class must be non-empty when computing keypoint losses.")

    if n_targets > 0 and target_classes.max() >= num_classes:
        logger.warning(
            "target_classes max index %d >= num_keypoints_per_class length %d; "
            "skipping keypoint loss for this batch to avoid crashing training. "
            "Check that your keypoint schema covers all annotation classes.",
            int(target_classes.max()),
            num_classes,
        )
        zeros = all_pred_keypoints.new_zeros(n_targets)
        return zeros, zeros, zeros, zeros

    if total_padded_num_keypoints % num_classes != 0:
        raise ValueError(
            f"total_padded_num_keypoints ({total_padded_num_keypoints}) must be divisible by"
            f" num_classes ({num_classes})"
        )
    kpad = total_padded_num_keypoints // num_classes
    split_pred_keypoints = all_pred_keypoints.view(n_targets, num_classes, kpad, pred_dim)
    selected_pred_keypoints = split_pred_keypoints[
        torch.arange(n_targets, device=split_pred_keypoints.device), target_classes
    ]

    active_keypoints_mask = torch.zeros(num_classes, kpad, dtype=torch.bool, device=split_pred_keypoints.device)
    for class_idx, num_keypoints in enumerate(num_keypoints_per_class):
        active_keypoints_mask[class_idx, :num_keypoints] = True

    keypoints_loss_mask = active_keypoints_mask[target_classes]
    keypoints_per_target = keypoints_loss_mask.sum(-1).to(dtype=selected_pred_keypoints.dtype)
    area = target_areas.to(torch.float32)
    area_eps = torch.finfo(area.dtype).eps
    valid_area = torch.isfinite(area) & (area > area_eps)
    valid_xy = torch.isfinite(selected_pred_keypoints[:, :, :2]).all(dim=-1) & torch.isfinite(
        target_keypoints[:, :, :2]
    ).all(dim=-1)
    valid_visibility = torch.isfinite(target_keypoints[:, :, 2]) & (target_keypoints[:, :, 2] > 0)
    location_loss_mask = keypoints_loss_mask & valid_visibility & valid_xy & valid_area.unsqueeze(1)
    location_count = location_loss_mask.sum(-1).to(dtype=selected_pred_keypoints.dtype)
    valid_count = location_count.clamp(min=1)
    denom_keypoints = keypoints_per_target.clamp(min=1).to(dtype=selected_pred_keypoints.dtype)
    safe_area_sqrt = area.clamp_min(area_eps).sqrt()

    scaled_masked_l1 = (
        F.l1_loss(selected_pred_keypoints[:, :, :2], target_keypoints[:, :, :2], reduction="none").sum(-1)
        * location_loss_mask.to(selected_pred_keypoints.dtype)
        / safe_area_sqrt.unsqueeze(1)
    )
    location_loss = scaled_masked_l1.sum(-1) / valid_count

    findable_loss = (
        F.binary_cross_entropy_with_logits(
            selected_pred_keypoints[:, :, 2],
            (target_keypoints[:, :, 2] > 0).to(selected_pred_keypoints.dtype),
            reduction="none",
        )
        * keypoints_loss_mask.to(selected_pred_keypoints.dtype)
    ).sum(-1) / denom_keypoints

    visible_loss = (
        F.binary_cross_entropy_with_logits(
            selected_pred_keypoints[:, :, 3],
            (target_keypoints[:, :, 2] > 1).to(selected_pred_keypoints.dtype),
            reduction="none",
        )
        * keypoints_loss_mask.to(selected_pred_keypoints.dtype)
    ).sum(-1) / denom_keypoints

    dxdy = (selected_pred_keypoints[:, :, :2] - target_keypoints[:, :, :2]).to(torch.float32)
    dx = dxdy[:, :, 0]
    dy = dxdy[:, :, 1]

    raw_log_l11 = selected_pred_keypoints[:, :, 4].to(torch.float32)
    raw_l21 = selected_pred_keypoints[:, :, 5].to(torch.float32)
    raw_log_l22 = selected_pred_keypoints[:, :, 6].to(torch.float32)
    finite_uncertainty = torch.isfinite(raw_log_l11) & torch.isfinite(raw_l21) & torch.isfinite(raw_log_l22)
    if not finite_uncertainty.all():
        logger.debug(
            "NLL loss: %d keypoint(s) with non-finite uncertainty dropped from loss.",
            (~finite_uncertainty).sum().item(),
        )
    gaussian_loss_mask = location_loss_mask & finite_uncertainty

    # Intentionally unclamped: the model is expected to learn bounded log-scale values;
    # clamping would mask divergence instead of exposing it during development.
    log_l11 = raw_log_l11
    l21 = raw_l21
    log_l22 = raw_log_l22

    l11 = log_l11.exp()
    l22 = log_l22.exp()
    u0 = l11 * dx + l21 * dy
    u1 = l22 * dy
    maha2 = u0 * u0 + u1 * u1
    gaussian_loss_mask = gaussian_loss_mask & torch.isfinite(u0) & torch.isfinite(u1) & torch.isfinite(maha2)
    gaussian_count = gaussian_loss_mask.sum(-1).to(dtype=selected_pred_keypoints.dtype)
    gaussian_valid_count = gaussian_count.clamp(min=1)
    nll_raw = 0.5 * (maha2 / area.clamp_min(area_eps).unsqueeze(1)) - (log_l11 + log_l22)
    nll_raw = torch.nan_to_num(
        nll_raw, nan=0.0, posinf=torch.finfo(nll_raw.dtype).max / 2, neginf=torch.finfo(nll_raw.dtype).min
    )
    nll_keypoints = nll_raw.masked_fill(~gaussian_loss_mask, 0.0)
    nll_loss = nll_keypoints.sum(-1) / gaussian_valid_count
    no_valid = gaussian_count <= 0
    nll_loss = torch.where(no_valid, torch.zeros_like(nll_loss), nll_loss)

    # NaN/Inf protection is upstream (see ``nan_to_num`` calls earlier in this function);
    # per-step hot-path assertions removed for performance (5–15% step latency).
    return location_loss, findable_loss, visible_loss, nll_loss


def _cdist_bce_with_logits(x: Tensor, y: Tensor) -> Tensor:
    """Compute pairwise BCE-with-logits summed along the last dim."""
    y_float = y.to(dtype=x.dtype)
    softplus = F.softplus(x).sum(dim=1, keepdim=True)
    dot = torch.matmul(x, y_float.t())
    return softplus - dot


def compute_keypoint_matching_cost(
    all_pred_keypoints: Tensor,
    target_keypoints: Tensor,
    target_classes: Tensor,
    target_areas: Tensor,
    num_keypoints_per_class: Sequence[int],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute many-to-many keypoint matching costs.

    Args:
        all_pred_keypoints: Predicted keypoints of shape ``(B, Q, K_total, >=7)``.
        target_keypoints: Ground truth keypoints with shape ``(N, Kmax, 3)``.
        target_classes: Class ids for each target with shape ``(N,)``.
        target_areas: Target box areas with shape ``(N,)``.
        num_keypoints_per_class: Number of keypoints per class.

    Returns:
        Tuple ``(cost_l1, cost_findable, cost_visible, cost_nll)`` each of shape ``(B, Q, N)``.

    Example:
        Compute matching costs for 1 image with 4 decoder queries against 2 ground-truth
        targets, each with 17 COCO keypoints:

        .. code-block:: python

            B, Q, K_total, pred_dim = 1, 4, 17, 7
            n_targets = 2

            all_pred_keypoints = torch.randn(B, Q, K_total, pred_dim)

            target_keypoints = torch.rand(n_targets, K_total, 3)
            target_keypoints[:, :, 2] = 2.0   # mark all keypoints fully visible

            target_classes = torch.zeros(n_targets, dtype=torch.long)  # single class (person)
            target_areas   = torch.tensor([0.05, 0.12])

            cost_l1, cost_findable, cost_visible, cost_nll = compute_keypoint_matching_cost(
                all_pred_keypoints,
                target_keypoints,
                target_classes,
                target_areas,
                num_keypoints_per_class=[17],
            )
            # Each output tensor has shape (1, 4, 2) — (B, Q, N).
            # cost_l1[:, q, n]  is the area-normalized L1 cost for query q against target n.
            # cost_nll[:, q, n] incorporates the Cholesky precision uncertainty.
            # These cost matrices are passed to the Hungarian matcher.
    """

    b, num_queries, total_num_keypoints, pred_dim = all_pred_keypoints.shape
    assert pred_dim >= 7, "Expected all_pred_keypoints last dim >= 7 (x,y,2 logits + 3 chol params)."
    n_targets = target_keypoints.shape[0]
    num_classes = len(num_keypoints_per_class)

    if num_classes == 0:
        raise ValueError("num_keypoints_per_class must be non-empty when computing keypoint matching costs.")

    if n_targets == 0:
        zeros = torch.zeros(
            (b, num_queries, n_targets),
            device=all_pred_keypoints.device,
            dtype=all_pred_keypoints.dtype,
        )
        return zeros, zeros, zeros, zeros

    kpad = total_num_keypoints // num_classes
    pred = all_pred_keypoints.view(b, num_queries, num_classes, kpad, pred_dim)

    cost_l1 = torch.zeros(
        (b, num_queries, n_targets),
        device=all_pred_keypoints.device,
        dtype=all_pred_keypoints.dtype,
    )
    cost_findable = torch.zeros(
        (b, num_queries, n_targets),
        device=all_pred_keypoints.device,
        dtype=all_pred_keypoints.dtype,
    )
    cost_visible = torch.zeros(
        (b, num_queries, n_targets),
        device=all_pred_keypoints.device,
        dtype=all_pred_keypoints.dtype,
    )
    cost_nll = torch.zeros(
        (b, num_queries, n_targets),
        device=all_pred_keypoints.device,
        dtype=all_pred_keypoints.dtype,
    )

    flat_bq = b * num_queries
    for class_idx in range(num_classes):
        target_indices = (target_classes == class_idx).nonzero().squeeze(1)
        if target_indices.numel() == 0:
            continue
        num_kpts = num_keypoints_per_class[class_idx]
        if num_kpts == 0:
            continue

        pred_by_class = pred[:, :, class_idx, :num_kpts, :]
        target_by_class = target_keypoints.index_select(0, target_indices)[:, :num_kpts, :]
        n_targets_by_class = target_by_class.shape[0]

        areas = target_areas.index_select(0, target_indices).to(torch.float32)
        area_eps = torch.finfo(areas.dtype).eps
        valid_area = torch.isfinite(areas) & (areas > area_eps)
        target_xy = target_by_class[:, :, :2]
        visible = (
            torch.isfinite(target_by_class[:, :, 2])
            & (target_by_class[:, :, 2] > 0)
            & torch.isfinite(target_xy).all(dim=-1)
            & valid_area.unsqueeze(1)
        )
        valid_per_target = visible.sum(dim=1).to(torch.float32)
        nll_denom = valid_per_target.clamp(min=1)
        has_visible = valid_per_target > 0

        area_sqrt = areas.clamp_min(area_eps).sqrt()

        # Vectorize over `num_kpts` — avoids one CUDA kernel launch per keypoint.
        # Shape conventions:
        #   pred_xy_flat:   (flat_bq, num_kpts, 2)
        #   target_xy_f32:  (n_targets_by_class, num_kpts, 2)
        #   visible:        (n_targets_by_class, num_kpts)
        #   diff tensors:   (flat_bq, n_targets_by_class, num_kpts)
        pred_xy_flat = pred_by_class[:, :, :, :2].reshape(flat_bq, num_kpts, 2).to(torch.float32)
        target_xy_f32 = target_xy.to(torch.float32)

        # L1 cost: (flat_bq, 1, num_kpts, 2) - (1, n_targets, num_kpts, 2) -> (flat_bq, n_targets, num_kpts, 2)
        diff = pred_xy_flat.unsqueeze(1) - target_xy_f32.unsqueeze(0)
        per_kpt_l1 = diff.abs().sum(-1)  # (flat_bq, n_targets, num_kpts)
        visible_btk = visible.unsqueeze(0)  # (1, n_targets, num_kpts)
        per_kpt_l1 = per_kpt_l1.masked_fill(~visible_btk, 0.0)
        scaled_loc = per_kpt_l1.sum(-1)  # (flat_bq, n_targets)

        loc_cost = (scaled_loc / nll_denom.unsqueeze(0)).div(area_sqrt.unsqueeze(0))
        loc_cost = torch.nan_to_num(loc_cost, nan=0.0, posinf=0.0, neginf=0.0)
        cost_l1[:, :, target_indices] = loc_cost.reshape(
            b,
            num_queries,
            n_targets_by_class,
        ).to(all_pred_keypoints.dtype)

        # NLL cost: Cholesky params (flat_bq, num_kpts) broadcast over targets axis.
        raw_log_l11 = pred_by_class[:, :, :, 4].reshape(flat_bq, num_kpts).to(torch.float32)
        raw_l21 = pred_by_class[:, :, :, 5].reshape(flat_bq, num_kpts).to(torch.float32)
        raw_log_l22 = pred_by_class[:, :, :, 6].reshape(flat_bq, num_kpts).to(torch.float32)
        # Intentionally unclamped: model is expected to learn bounded log-scale values.
        log_l11 = raw_log_l11
        l21 = raw_l21
        log_l22 = raw_log_l22
        l11 = log_l11.exp()  # (flat_bq, num_kpts)
        l22 = log_l22.exp()

        finite_xy = torch.isfinite(pred_xy_flat).all(dim=-1)  # (flat_bq, num_kpts)
        finite_pred = finite_xy & torch.isfinite(raw_log_l11) & torch.isfinite(raw_l21) & torch.isfinite(raw_log_l22)

        dx = diff[..., 0]  # (flat_bq, n_targets, num_kpts)
        dy = diff[..., 1]
        u0 = l11.unsqueeze(1) * dx + l21.unsqueeze(1) * dy
        u1 = l22.unsqueeze(1) * dy
        maha2 = u0 * u0 + u1 * u1

        keypoint_mask = (
            visible_btk & finite_pred.unsqueeze(1) & torch.isfinite(u0) & torch.isfinite(u1) & torch.isfinite(maha2)
        )
        nll_k = 0.5 * (maha2 / areas.clamp_min(area_eps).view(1, n_targets_by_class, 1)) - (
            log_l11 + log_l22
        ).unsqueeze(1)
        nll_k = torch.nan_to_num(
            nll_k, nan=0.0, posinf=torch.finfo(nll_k.dtype).max / 2, neginf=torch.finfo(nll_k.dtype).min
        )
        nll_k = nll_k.masked_fill(~keypoint_mask, 0.0)
        nll_sum = nll_k.sum(-1)  # (flat_bq, n_targets)

        mean_nll = (nll_sum / nll_denom.unsqueeze(0)).reshape(b, num_queries, n_targets_by_class)
        mean_nll.masked_fill_(~has_visible.unsqueeze(0), 0.0)
        cost_nll[:, :, target_indices] = mean_nll.to(all_pred_keypoints.dtype)

        pred_findable = pred_by_class[:, :, :, 2].reshape(flat_bq, num_kpts)
        target_findable = (
            (target_by_class[:, :, 2] > 0)
            .to(all_pred_keypoints.dtype)
            .reshape(
                n_targets_by_class,
                num_kpts,
            )
        )
        pred_visible = pred_by_class[:, :, :, 3].reshape(flat_bq, num_kpts)
        target_visible = (
            (target_by_class[:, :, 2] > 1)
            .to(all_pred_keypoints.dtype)
            .reshape(
                n_targets_by_class,
                num_kpts,
            )
        )
        cost_findable[:, :, target_indices] = _cdist_bce_with_logits(pred_findable, target_findable).reshape(
            b,
            num_queries,
            n_targets_by_class,
        ) / float(num_kpts)
        cost_visible[:, :, target_indices] = _cdist_bce_with_logits(pred_visible, target_visible).reshape(
            b,
            num_queries,
            n_targets_by_class,
        ) / float(num_kpts)

    # NaN/Inf protection is upstream (see ``nan_to_num`` calls earlier); per-step
    # matching-cost assertions removed for performance (5–15% step latency).
    return cost_l1, cost_findable, cost_visible, cost_nll


__all__ = [
    "ConditionalQueryInitializer",
    "compute_keypoint_matching_cost",
    "compute_l1_keypoint_loss",
]

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
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
"""Modules to compute the matching cost and solve the corresponding LSAP."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment as _linear_sum_assignment  # type: ignore[import-untyped,unused-ignore]
from torch import Tensor, nn

from mmdet.models.layers.rf_detr.models.heads.keypoints import compute_keypoint_matching_cost
from mmdet.models.layers.rf_detr.models.heads.segmentation import point_sample
from mmdet.models.layers.rf_detr.utilities.box_ops import batch_dice_loss, batch_sigmoid_ce_loss, box_cxcywh_to_xyxy, generalized_box_iou
from mmdet.models.layers.rf_detr.utilities.logger import get_logger

logger = get_logger()
_SANITIZED_COST_MARGIN = 1.0
_FOCAL_LOSS_GAMMA = 2.0
_LinearSumAssignment = Callable[[Any], tuple[NDArray[np.int64], NDArray[np.int64]]]
linear_sum_assignment = cast(_LinearSumAssignment, _linear_sum_assignment)


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network For efficiency reasons,
    the targets don't include the no_object.

    Because of this, in general, there are more predictions than targets. In this case, we do a 1-to-1 matching of the
    best predictions, while the others are un-matched (and thus treated as non-objects).

    Note:
        The focal loss exponent ``gamma`` is fixed at ``_FOCAL_LOSS_GAMMA`` (2.0) and is not
        configurable. Only ``focal_alpha`` can be adjusted at construction time.
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_alpha: float = 0.25,
        use_pos_only: bool = False,  # reserved for future use; not yet implemented
        use_position_modulated_cost: bool = False,  # reserved for future use; not yet implemented
        mask_point_sample_ratio: int = 16,
        cost_mask_ce: float = 1,
        cost_mask_dice: float = 1,
        num_keypoints_per_class: list[int] | None = None,
        keypoint_l1_loss_coef: float = 0.0,
        keypoint_findable_loss_coef: float = 0.0,
        keypoint_visible_loss_coef: float = 0.0,
        keypoint_nll_loss_coef: float = 0.0,
    ):
        """Creates the matcher.

        Args:
            cost_class: Relative weight of the classification error in the matching cost.
            cost_bbox: Relative weight of the L1 error of the bounding box coordinates.
            cost_giou: Relative weight of the GIoU loss of the bounding box.
            focal_alpha: Alpha parameter for focal loss used in the classification cost.
            use_pos_only: Reserved for future use; currently has no effect.
            use_position_modulated_cost: Reserved for future use; currently has no effect.
            mask_point_sample_ratio: Downsampling ratio for mask point sampling.
            cost_mask_ce: Relative weight of the binary cross-entropy mask cost.
            cost_mask_dice: Relative weight of the Dice mask cost.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs can't be 0"
        self.focal_alpha = focal_alpha
        self.mask_point_sample_ratio = mask_point_sample_ratio
        self.cost_mask_ce = cost_mask_ce
        self.cost_mask_dice = cost_mask_dice
        self.num_keypoints_per_class = num_keypoints_per_class or []
        self.keypoint_l1_loss_coef = keypoint_l1_loss_coef
        self.keypoint_findable_loss_coef = keypoint_findable_loss_coef
        self.keypoint_visible_loss_coef = keypoint_visible_loss_coef
        self.keypoint_nll_loss_coef = keypoint_nll_loss_coef
        self._warned_non_finite_costs = False

    @staticmethod
    def _sanitize_cost_matrix(cost_matrix: Tensor) -> Tensor:
        """Replace non-finite cost entries with a large finite sentinel.

        >>> HungarianMatcher._sanitize_cost_matrix(
        ...     torch.tensor([[1.0, float("nan")], [float("inf"), -2.0]])
        ... ).tolist()
        [[1.0, 4.0], [4.0, -2.0]]

        Args:
            cost_matrix: Cost matrix to sanitize before Hungarian assignment.

        Returns:
            Cost matrix with all non-finite entries replaced by a finite sentinel that is no smaller than any valid
            entry.
        """
        finite_mask = torch.isfinite(cost_matrix)
        if finite_mask.all():
            return cost_matrix

        dtype_info = torch.finfo(cost_matrix.dtype)
        if finite_mask.any():
            finite_costs = cost_matrix[finite_mask]
            max_cost = finite_costs.max()
            # Add the largest absolute finite cost so the replacement stays
            # strictly larger than every valid entry, even if all costs are negative.
            replacement_cost = max_cost + finite_costs.abs().max() + _SANITIZED_COST_MARGIN
            # Guard against overflow to inf/NaN and clamp to the maximum finite value.
            if not torch.isfinite(replacement_cost):
                replacement_cost = cost_matrix.new_tensor(dtype_info.max)
            else:
                replacement_cost = torch.clamp(replacement_cost, max=dtype_info.max)
        else:
            # If all entries are non-finite, fall back to a large finite sentinel.
            replacement_cost = cost_matrix.new_tensor(dtype_info.max)

        sanitized_cost_matrix = cost_matrix.clone()
        sanitized_cost_matrix[~finite_mask] = replacement_cost
        return sanitized_cost_matrix

    @torch.no_grad()
    def forward(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Any]],
        group_detr: int = 1,
    ) -> list[tuple[Tensor, Tensor]]:
        """Performs the matching

        Args:
            outputs: Dict containing at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: List of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates "masks": Tensor of
                 dim [num_target_boxes, H, W] containing the target mask coordinates
            group_detr: Number of groups used for matching.

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds len(index_i) == len(index_j). With group_detr == 1 this
            length is min(num_queries, num_target_boxes); with group_detr > 1 the per-group matches are
            concatenated, so the length is that quantity summed over the groups.
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        flat_pred_logits = outputs["pred_logits"].flatten(0, 1)
        out_prob = flat_pred_logits.sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_keypoints = None

        masks_present = "masks" in targets[0]
        keypoints_present = "pred_keypoints" in outputs and "keypoints" in targets[0]
        if keypoints_present:
            tgt_keypoints = torch.cat([v["keypoints"] for v in targets], dim=0)

        # Compute the giou cost between boxes
        giou = generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = -giou

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = _FOCAL_LOSS_GAMMA

        # neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        # pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        # we refactor these with logsigmoid for numerical stability
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-F.logsigmoid(-flat_pred_logits))
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-F.logsigmoid(flat_pred_logits))
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        if masks_present:
            tgt_masks = torch.cat([v["masks"] for v in targets])

            if isinstance(outputs["pred_masks"], Tensor):
                out_masks = outputs["pred_masks"].flatten(0, 1)

                num_points = out_masks.shape[-2] * out_masks.shape[-1] // self.mask_point_sample_ratio

                point_coords = torch.rand(1, num_points, 2, device=out_masks.device)
                pred_masks_logits = point_sample(
                    out_masks.unsqueeze(1), point_coords.repeat(out_masks.shape[0], 1, 1), align_corners=False
                ).squeeze(1)
            else:
                spatial_features = outputs["pred_masks"]["spatial_features"]
                query_features = outputs["pred_masks"]["query_features"]
                bias = outputs["pred_masks"]["bias"]

                num_points = spatial_features.shape[-2] * spatial_features.shape[-1] // self.mask_point_sample_ratio
                point_coords = torch.rand(1, num_points, 2, device=spatial_features.device)
                pred_masks_logits = point_sample(
                    spatial_features, point_coords.repeat(spatial_features.shape[0], 1, 1), align_corners=False
                )
                # print(f"pred_masks_logits.shape: {pred_masks_logits.shape}")
                pred_masks_logits = torch.einsum("bcp,bnc->bnp", pred_masks_logits, query_features) + bias
                pred_masks_logits = pred_masks_logits.flatten(0, 1)

            tgt_masks = tgt_masks.to(pred_masks_logits.dtype)
            tgt_masks_flat = point_sample(
                tgt_masks.unsqueeze(1),
                point_coords.repeat(tgt_masks.shape[0], 1, 1),
                align_corners=False,
                mode="nearest",
            ).squeeze(1)

            # Binary cross-entropy with logits cost (mean over pixels), computed pairwise efficiently
            cost_mask_ce = batch_sigmoid_ce_loss(pred_masks_logits, tgt_masks_flat)

            # Dice loss cost (1 - dice coefficient)
            cost_mask_dice = batch_dice_loss(pred_masks_logits, tgt_masks_flat)

        if keypoints_present and tgt_keypoints is not None:
            target_areas = tgt_bbox[:, 2] * tgt_bbox[:, 3]
            cost_l1, cost_findable, cost_visible, cost_nll = compute_keypoint_matching_cost(
                all_pred_keypoints=outputs["pred_keypoints"],
                target_keypoints=tgt_keypoints,
                target_classes=tgt_ids,
                target_areas=target_areas,
                num_keypoints_per_class=self.num_keypoints_per_class,
            )
            cost_l1 = cost_l1.flatten(0, 1)
            cost_findable = cost_findable.flatten(0, 1)
            cost_visible = cost_visible.flatten(0, 1)
            cost_nll = cost_nll.flatten(0, 1)

        # Final cost matrix
        cost_matrix = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        if masks_present:
            cost_matrix = cost_matrix + self.cost_mask_ce * cost_mask_ce + self.cost_mask_dice * cost_mask_dice
        if keypoints_present:
            cost_matrix = (
                cost_matrix
                + self.keypoint_l1_loss_coef * cost_l1
                + self.keypoint_findable_loss_coef * cost_findable
                + self.keypoint_visible_loss_coef * cost_visible
                + self.keypoint_nll_loss_coef * cost_nll
            )
        cost_matrix = (
            cost_matrix.view(bs, num_queries, -1).float().cpu()
        )  # convert to float because bfloat16 doesn't play nicely with CPU

        # We assume any good match will not cause NaN or Inf, so replace invalid
        # entries with a finite value that is larger than every valid cost.
        finite_mask = torch.isfinite(cost_matrix)
        if not finite_mask.all():
            if not self._warned_non_finite_costs:
                logger.warning(
                    "Non-finite values detected in matcher cost matrix; "
                    "replacing with finite sentinel. "
                    "Check for numerical instability."
                )
                self._warned_non_finite_costs = True
            cost_matrix = self._sanitize_cost_matrix(cost_matrix)

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        if num_queries % group_detr != 0:
            raise ValueError(f"num_queries ({num_queries}) must be divisible by group_detr ({group_detr})")
        g_num_queries = num_queries // group_detr
        cost_matrix_list = cost_matrix.split(g_num_queries, dim=1)
        for g_i in range(group_detr):
            grouped_cost_matrix = cost_matrix_list[g_i]
            indices_g = [linear_sum_assignment(c[i]) for i, c in enumerate(grouped_cost_matrix.split(sizes, -1))]
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (
                        np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]),
                        np.concatenate([indice1[1], indice2[1]]),
                    )
                    for indice1, indice2 in zip(indices, indices_g)
                ]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args: Any) -> HungarianMatcher:
    """Build a HungarianMatcher from a training argument namespace.

    Args:
        args: Namespace supplying ``focal_alpha``, ``set_cost_class``, ``set_cost_bbox``,
            ``set_cost_giou``, ``segmentation_head``, and optional keypoint cost
            coefficients (``keypoint_l1_loss_coef``, ``keypoint_findable_loss_coef``,
            ``keypoint_visible_loss_coef``, ``keypoint_nll_loss_coef``). When
            ``segmentation_head`` is truthy, also requires ``mask_ce_loss_coef``,
            ``mask_dice_loss_coef``, and ``mask_point_sample_ratio``.

    Returns:
        Configured HungarianMatcher instance.
    """
    # Detection-only matcher args may omit keypoint costs; zero defaults disable keypoint matching terms.
    common_kwargs = {
        "cost_class": args.set_cost_class,
        "cost_bbox": args.set_cost_bbox,
        "cost_giou": args.set_cost_giou,
        "focal_alpha": args.focal_alpha,
        "num_keypoints_per_class": getattr(args, "num_keypoints_per_class", []),
        "keypoint_l1_loss_coef": getattr(args, "keypoint_l1_loss_coef", 0.0),
        "keypoint_findable_loss_coef": getattr(args, "keypoint_findable_loss_coef", 0.0),
        "keypoint_visible_loss_coef": getattr(args, "keypoint_visible_loss_coef", 0.0),
        "keypoint_nll_loss_coef": getattr(args, "keypoint_nll_loss_coef", 0.0),
    }
    if args.segmentation_head:
        return HungarianMatcher(
            **common_kwargs,
            cost_mask_ce=args.mask_ce_loss_coef,
            cost_mask_dice=args.mask_dice_loss_coef,
            mask_point_sample_ratio=args.mask_point_sample_ratio,
        )
    else:
        return HungarianMatcher(**common_kwargs)

# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""Greedy matching and accumulation functions for evaluation metrics."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torchvision.ops import box_iou

from mmdet.models.layers.rf_detr.utilities import all_gather


def _compute_mask_iou(pred_masks: Tensor, gt_masks: Tensor) -> Tensor:
    """Compute pairwise boolean-mask IoU between N predictions and M ground truths.

    Args:
        pred_masks: Boolean mask tensor of shape [N, H, W].
        gt_masks: Boolean mask tensor of shape [M, H, W].

    Returns:
        IoU tensor of shape [N, M].
    """
    n = pred_masks.shape[0]
    m = gt_masks.shape[0]
    if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
        h, w = pred_masks.shape[-2:]
        gt_masks = F.interpolate(gt_masks.float().unsqueeze(1), size=(h, w), mode="nearest").squeeze(1)
    pred_flat = pred_masks.bool().view(n, -1).float()  # [N, HW]
    gt_flat = gt_masks.bool().view(m, -1).float()  # [M, HW]
    inter = torch.mm(pred_flat, gt_flat.t())  # [N, M]
    pred_area = pred_flat.sum(dim=1, keepdim=True)  # [N, 1]
    gt_area = gt_flat.sum(dim=1, keepdim=True)  # [M, 1]
    union = pred_area + gt_area.t() - inter  # [N, M]
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


def _match_single_class(
    pred_scores: Tensor,
    pred_items: Tensor,
    gt_items: Tensor,
    gt_crowd: Tensor,
    iou_threshold: float,
    iou_type: str,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float32]],
    np.ndarray[Any, np.dtype[np.int64]],
    np.ndarray[Any, np.dtype[np.bool_]],
    int,
]:
    """Greedy highest-score-first matching for one class in one image.

    Implements the COCO matching algorithm: each GT is matched at most once; detections are processed in descending
    score order; detections matched to crowd GTs are marked as ignored rather than false positives.

    Args:
        pred_scores: Float tensor of shape [N] with detection confidences.
        pred_items: Predictions — boxes [N, 4] in xyxy coords or masks [N, H, W].
        gt_items: Ground truths — boxes [M, 4] in xyxy coords or masks [M, H, W].
        gt_crowd: Bool tensor of shape [M], True for crowd instances.
        iou_threshold: Minimum IoU to count as a positive match.
        iou_type: ``"bbox"`` for box IoU or ``"segm"`` for mask IoU.

    Returns:
        Tuple ``(scores_np, matches_np, ignore_np, total_gt)`` where:
            - scores_np: float32 array [N] ordered by descending score.
            - matches_np: int array [N], 1 = TP, 0 = FP.
            - ignore_np: bool array [N], True if matched to a crowd GT.
            - total_gt: number of non-crowd GT instances.
    """
    n = pred_scores.shape[0]
    m = gt_items.shape[0]

    sort_idx = torch.argsort(pred_scores, descending=True)
    pred_scores_sorted = pred_scores[sort_idx]
    pred_sorted = pred_items[sort_idx]

    if iou_type == "bbox":
        iou_matrix = box_iou(pred_sorted, gt_items)  # [N, M]
    else:
        iou_matrix = _compute_mask_iou(pred_sorted, gt_items)  # [N, M]

    device = pred_scores.device
    gt_matched = torch.zeros(m, dtype=torch.bool, device=device)
    pred_match = torch.zeros(n, dtype=torch.long, device=device)
    pred_ignore = torch.zeros(n, dtype=torch.bool, device=device)

    for i in range(n):
        ious = iou_matrix[i]  # [M]

        # Try to match to a non-crowd GT (each non-crowd GT matched at most once).
        nc_ious = ious.clone()
        nc_ious[gt_crowd] = -1.0
        nc_ious[gt_matched & ~gt_crowd] = -1.0  # already claimed

        best_nc_iou, best_nc_idx = nc_ious.max(dim=0)
        if best_nc_iou >= iou_threshold:
            pred_match[i] = 1
            gt_matched[best_nc_idx] = True
        # A detection matched to a crowd GT is ignored (not a false positive).
        elif gt_crowd.any():
            crowd_ious = ious.clone()
            crowd_ious[~gt_crowd] = -1.0
            if crowd_ious.max() >= iou_threshold:
                pred_ignore[i] = True
            # else: false positive — pred_match stays 0

    total_gt = int((~gt_crowd).sum().item())
    return (
        np.asarray(pred_scores_sorted.float().cpu().numpy(), dtype=np.float32),
        np.asarray(pred_match.cpu().numpy(), dtype=np.int64),
        np.asarray(pred_ignore.cpu().numpy(), dtype=np.bool_),
        total_gt,
    )


def build_matching_data(
    preds_list: list[dict[str, Tensor]],
    targets_list: list[dict[str, Tensor]],
    iou_threshold: float = 0.5,
    iou_type: str = "bbox",
) -> dict[int, dict[str, Any]]:
    """Build compact per-class matching data from a batch of predictions and targets.

    Implements greedy highest-score-first matching compatible with the COCO algorithm. The returned dict can be passed
    directly to ``merge_matching_data()`` and ultimately consumed by ``sweep_confidence_thresholds()`` after conversion
    to list form.

    Args:
        preds_list: Per-image predictions. Each dict must contain:

            - ``boxes``: float Tensor [N, 4] in absolute xyxy coordinates.
            - ``scores``: float Tensor [N].
            - ``labels``: int64 Tensor [N].
            - ``masks`` *(optional)*: bool Tensor [N, H, W] for segmentation.

        targets_list: Per-image ground truths. Each dict must contain:

            - ``boxes``: float Tensor [M, 4] in absolute xyxy coordinates.
            - ``labels``: int64 Tensor [M].
            - ``masks`` *(optional)*: bool Tensor [M, H, W] for segmentation.
            - ``iscrowd`` *(optional)*: int64 Tensor [M], 1 for crowd instances.

        iou_threshold: IoU threshold for positive matching. Defaults to 0.5.
        iou_type: ``"bbox"`` for bounding-box IoU; ``"segm"`` for boolean-mask
            IoU. Defaults to ``"bbox"``.

    Returns:
        Dict mapping ``class_id`` (int) to a compact matching dict with keys:

            - ``"scores"``: float32 ndarray of detection scores.
            - ``"matches"``: int ndarray (1 = TP, 0 = FP).
            - ``"ignore"``: bool ndarray (True if matched to a crowd GT).
            - ``"total_gt"``: int, count of non-crowd GT instances.
    """
    acc: dict[int, dict[str, list[Any] | int]] = {}

    for preds, targets in zip(preds_list, targets_list):
        pred_boxes = preds["boxes"]  # [N, 4]
        pred_scores = preds["scores"]  # [N]
        pred_labels = preds["labels"]  # [N]
        pred_masks = preds.get("masks")  # [N, H, W] | None

        gt_boxes = targets["boxes"]  # [M, 4]
        gt_labels = targets["labels"]  # [M]
        gt_masks = targets.get("masks")  # [M, H, W] | None
        raw_crowd = targets.get(
            "iscrowd",
            torch.zeros(len(gt_labels), dtype=torch.long, device=gt_labels.device),
        )
        gt_crowd = raw_crowd.bool()

        gt_label_ids = cast(list[int], gt_labels.tolist())
        pred_label_ids = cast(list[int], pred_labels.tolist())
        all_class_ids: set[int] = set(gt_label_ids) | set(pred_label_ids)

        for class_id in all_class_ids:
            pred_mask_c = pred_labels == class_id
            gt_mask_c = gt_labels == class_id

            p_scores = pred_scores[pred_mask_c]
            gt_crowd_c = gt_crowd[gt_mask_c]
            n_pred = int(pred_mask_c.sum().item())
            n_gt = int(gt_mask_c.sum().item())

            entry = acc.setdefault(
                class_id,
                {"scores": [], "matches": [], "ignore": [], "total_gt": 0},
            )

            if n_pred == 0:
                entry["total_gt"] = cast(int, entry["total_gt"]) + int((~gt_crowd_c).sum().item())
                continue

            if n_gt == 0:
                # TODO: support bfloat16 natively once numpy adds bf16 dtype
                sc = np.asarray(p_scores.float().cpu().numpy(), dtype=np.float32)
                order = np.argsort(-sc)
                cast(list[float], entry["scores"]).extend(sc[order].tolist())
                cast(list[int], entry["matches"]).extend([0] * n_pred)
                cast(list[bool], entry["ignore"]).extend([False] * n_pred)
                continue

            if iou_type == "bbox":
                p_items: Tensor = pred_boxes[pred_mask_c]  # [n_pred, 4]
                gt_items: Tensor = gt_boxes[gt_mask_c]  # [n_gt, 4]
            else:
                if pred_masks is None or gt_masks is None:
                    raise ValueError("iou_type='segm' requires 'masks' in both preds and targets")
                p_items = pred_masks[pred_mask_c]  # [n_pred, H, W]
                gt_items = gt_masks[gt_mask_c]  # [n_gt, H, W]

            scores_np, matches_np, ignore_np, total_gt = _match_single_class(
                p_scores, p_items, gt_items, gt_crowd_c, iou_threshold, iou_type
            )

            cast(list[float], entry["scores"]).extend(float(score) for score in scores_np)
            cast(list[int], entry["matches"]).extend(int(match) for match in matches_np)
            cast(list[bool], entry["ignore"]).extend(bool(ignore) for ignore in ignore_np)
            entry["total_gt"] = cast(int, entry["total_gt"]) + total_gt

    return {
        class_id: {
            "scores": np.array(data["scores"], dtype=np.float32),
            "matches": np.array(data["matches"], dtype=np.int64),
            "ignore": np.array(data["ignore"], dtype=np.bool_),
            "total_gt": data["total_gt"],
        }
        for class_id, data in acc.items()
    }


def init_matching_accumulator() -> dict[int, dict[str, Any]]:
    """Return an empty matching accumulator compatible with ``merge_matching_data()``.

    Returns:
        Empty dict to be passed as the first argument to ``merge_matching_data()``.
    """
    return {}


def merge_matching_data(
    accumulator: dict[int, dict[str, Any]],
    new_data: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Merge *new_data* into *accumulator* in place.

    Both arguments share the dict schema produced by ``build_matching_data()``: each class-keyed sub-dict contains
    ``"scores"`` (float32 ndarray), ``"matches"`` (int64 ndarray), ``"ignore"`` (bool ndarray), and ``"total_gt"``
    (int).

    Args:
        accumulator: Running accumulator, modified in place.
        new_data: Batch-level matching data to merge in.

    Returns:
        The modified *accumulator* (same object, for method chaining).
    """
    for class_id, data in new_data.items():
        if class_id not in accumulator:
            accumulator[class_id] = {
                "scores": data["scores"].copy(),
                "matches": data["matches"].copy(),
                "ignore": data["ignore"].copy(),
                "total_gt": data["total_gt"],
            }
        else:
            entry = accumulator[class_id]
            entry["scores"] = np.concatenate([entry["scores"], data["scores"]])
            entry["matches"] = np.concatenate([entry["matches"], data["matches"]])
            entry["ignore"] = np.concatenate([entry["ignore"], data["ignore"]])
            entry["total_gt"] += data["total_gt"]
    return accumulator


def distributed_merge_matching_data(
    local_data: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Gather per-rank matching data from all DDP ranks and merge into one dict.

    Uses ``rfdetr.utilities.all_gather`` (pickle-based) so the data need not be a tensor. In single-process
    (non-distributed) mode, returns a merged copy of *local_data* unchanged.

    Args:
        local_data: Per-rank accumulator produced by ``merge_matching_data()``.

    Returns:
        Merged accumulator containing contributions from all ranks.
    """
    gathered: list[dict[int, dict[str, Any]]] = all_gather(local_data)
    merged: dict[int, dict[str, Any]] = {}
    for rank_data in gathered:
        merge_matching_data(merged, rank_data)
    return merged

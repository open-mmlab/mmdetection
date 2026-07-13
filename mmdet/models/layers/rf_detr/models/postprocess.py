# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Extracted from lwdetr.py (Phase 10)
# Original copyrights: LW-DETR (Baidu), Conditional DETR (Microsoft),
# DETR (Facebook), Deformable DETR (SenseTime)
# ------------------------------------------------------------------------
"""Post-processing module for converting model outputs to COCO API format."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from mmdet.models.layers.rf_detr.utilities import box_ops

# Number of masks upsampled per interpolation call. Bounds peak memory when many masks are
# resized to full image resolution (e.g. K=300 at 1080p would otherwise allocate gigabytes).
_MASK_CHUNK = 32


class PostProcess(nn.Module):
    """Convert raw RF-DETR model outputs into per-image prediction tensors.

    The postprocessor is shared by detection, segmentation, and keypoint inference. It selects top scoring query/class
    pairs, scales boxes back to the requested image sizes, and then delegates to the head-specific private helper for
    masks, keypoints, or box-only results.
    """

    def __init__(
        self,
        num_select: int = 300,
        num_keypoints_per_class: list[int] | None = None,
        trace_alpha: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_select = num_select
        self.num_keypoints_per_class = num_keypoints_per_class or []
        self.trace_alpha = trace_alpha

    @torch.no_grad()
    def forward(self, outputs: dict[str, torch.Tensor], target_sizes: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Convert raw model tensors into per-image detection dictionaries.

        Args:
            outputs: Model output dictionary containing ``pred_logits`` and ``pred_boxes`` plus optional
                ``pred_masks`` or ``pred_keypoints``.
            target_sizes: Per-image ``(height, width)`` tensor. For inference and evaluation this should be the
                original image size so normalized boxes and keypoints are returned in source-image pixel coordinates.

        Returns:
            One dictionary per image. Every dictionary contains ``scores``, ``labels``, and ``boxes`` in absolute
            pixel coordinates clamped to the respective image dimensions. Segmentation outputs also contain
            ``masks``. Keypoint outputs also contain ``keypoints`` and ``keypoint_precision_cholesky``.
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        out_masks = outputs.get("pred_masks")
        out_keypoints = outputs.get("pred_keypoints")
        self._validate_outputs(out_logits, out_masks, out_keypoints, target_sizes)

        scores, labels, topk_boxes = self._select_topk(out_logits)
        boxes = self._gather_and_scale_boxes(out_bbox, topk_boxes, target_sizes)

        if out_masks is not None:
            return self._postprocess_masks(out_masks, scores, labels, boxes, topk_boxes, target_sizes)
        if out_keypoints is not None:
            return self._postprocess_keypoints(out_keypoints, scores, labels, boxes, topk_boxes, target_sizes)
        return self._postprocess_boxes(scores, labels, boxes)

    @staticmethod
    def _validate_outputs(
        out_logits: torch.Tensor,
        out_masks: torch.Tensor | None,
        out_keypoints: torch.Tensor | None,
        target_sizes: torch.Tensor,
    ) -> None:
        """Validate mutually exclusive output heads and per-image target sizes.

        Args:
            out_logits: Classification logits with shape ``(B, Q, C)``.
            out_masks: Optional mask logits from segmentation models.
            out_keypoints: Optional keypoint predictions from keypoint models.
            target_sizes: Per-image ``(height, width)`` tensor with shape
                ``(B, 2)``.

        Raises:
            ValueError: If both masks and keypoints are present in the model
                outputs at the same time. Mask and keypoint heads are mutually
                exclusive at inference.
            AssertionError: If batch dimensions do not match ``target_sizes``.
        """
        if out_masks is not None and out_keypoints is not None:
            raise ValueError("masks and keypoints cannot be used together in postprocessing.")
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

    def _select_topk(self, out_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select the highest scoring query/class pairs.

        Args:
            out_logits: Classification logits with shape ``(B, Q, C)``.

        Returns:
            Tuple containing selected scores, class labels, and query indices.
            Scores are sigmoid probabilities before any keypoint uncertainty
            fusion. Labels are class indices and query indices select rows from
            box/mask/keypoint outputs.
        """
        prob = out_logits.sigmoid()
        logits_for_topk = prob.view(out_logits.shape[0], -1)
        num_to_select = min(self.num_select, logits_for_topk.shape[1])
        topk_values, topk_indexes = torch.topk(logits_for_topk, num_to_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        return scores, labels, topk_boxes

    @staticmethod
    def _gather_and_scale_boxes(
        out_bbox: torch.Tensor,
        topk_boxes: torch.Tensor,
        target_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """Gather selected boxes and scale normalized coordinates to pixels.

        Args:
            out_bbox: Normalized ``cxcywh`` boxes with shape ``(B, Q, 4)``.
            topk_boxes: Query indices selected by :meth:`_select_topk`.
            target_sizes: Per-image ``(height, width)`` tensor.

        Returns:
            Absolute ``xyxy`` boxes with shape ``(B, K, 4)`` in pixel units,
            clamped to ``[0, width]`` for x-coordinates and ``[0, height]`` for y-coordinates.
        """
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.dtype)
        boxes = boxes * scale_fct[:, None, :]
        # Model regression is unbounded; clamp to valid pixel range before returning.
        boxes = boxes.clamp_min(0.0).clamp(max=scale_fct[:, None, :])
        return boxes

    @staticmethod
    def _postprocess_masks(
        out_masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        boxes: torch.Tensor,
        topk_boxes: torch.Tensor,
        target_sizes: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Attach resized segmentation masks for selected detections.

        Args:
            out_masks: Raw mask logits with shape ``(B, Q, Hm, Wm)``.
            scores: Selected object scores with shape ``(B, K)``.
            labels: Selected class labels with shape ``(B, K)``.
            boxes: Selected absolute boxes with shape ``(B, K, 4)``.
            topk_boxes: Selected query indices with shape ``(B, K)``.
            target_sizes: Per-image ``(height, width)`` tensor used for mask
                resizing.

        Returns:
            One result dict per image containing scores, labels, boxes, and
            boolean masks resized to the target image size.
        """
        results = []
        for i in range(out_masks.shape[0]):
            res_i = {"scores": scores[i], "labels": labels[i], "boxes": boxes[i]}
            k_idx = topk_boxes[i]
            masks_i = torch.gather(
                out_masks[i],
                0,
                k_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, out_masks.shape[-2], out_masks.shape[-1]),
            )  # [K, Hm, Wm]
            h, w = target_sizes[i].tolist()
            # Upsample in chunks and threshold *inside* the comprehension so only one float32 chunk
            # is live at a time; the accumulated list holds bool tensors (1 byte/pixel vs 4 for float32).
            # At K=300, 1080p this reduces peak memory from ~5 GB to ~1.5 GB vs a single F.interpolate
            # call that would allocate the full [K,1,H,W] float tensor followed by a bool copy.
            chunks = [
                F.interpolate(
                    masks_i[start : start + _MASK_CHUNK].unsqueeze(1),
                    size=(int(h), int(w)),
                    mode="bilinear",
                    align_corners=False,
                )
                > 0.0
                for start in range(0, masks_i.shape[0], _MASK_CHUNK)
            ]
            masks_i = (
                torch.cat(chunks, dim=0) if chunks else masks_i.new_zeros((0, 1, int(h), int(w)), dtype=torch.bool)
            )  # [K,1,H,W] bool
            res_i["masks"] = masks_i
            results.append(res_i)
        return results

    def _postprocess_keypoints(
        self,
        out_keypoints: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        boxes: torch.Tensor,
        topk_boxes: torch.Tensor,
        target_sizes: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Select class-specific keypoints and optionally fuse object scores.

        Args:
            out_keypoints: Raw keypoint predictions with shape
                ``(B, Q, C * max(K_c), D)`` where keypoint slots are padded per
                class.
            scores: Selected object scores before uncertainty fusion.
            labels: Selected class labels.
            boxes: Selected absolute boxes.
            topk_boxes: Selected query indices.
            target_sizes: Per-image ``(height, width)`` tensor.

        Returns:
            One result dict per image containing postprocessed object scores,
            labels, boxes, pixel-space keypoints, and raw precision-Cholesky
            parameters. When ``trace_alpha > 0``, scores for valid keypoint
            classes are fused with uncertainty from the active keypoints of the
            predicted class and normalized to ``[0, 1)``.
        """
        results = []
        max_num_keypoints = max(self.num_keypoints_per_class, default=0)
        num_keypoint_classes = len(self.num_keypoints_per_class)
        for i in range(out_keypoints.shape[0]):
            labels_i = labels[i]
            boxes_i = boxes[i]
            scores_i = scores[i]
            keypoints_i = self._gather_keypoints_for_queries(out_keypoints[i], topk_boxes[i])
            output_keypoints, output_keypoint_precision = self._empty_keypoint_outputs(
                keypoints_i,
                max_num_keypoints,
            )
            if num_keypoint_classes > 0 and max_num_keypoints > 0:
                scores_i, output_keypoints, output_keypoint_precision = self._decode_keypoints_for_image(
                    keypoints_i=keypoints_i,
                    labels_i=labels_i,
                    scores_i=scores_i,
                    target_size=target_sizes[i],
                    output_keypoints=output_keypoints,
                    output_keypoint_precision=output_keypoint_precision,
                    num_keypoint_classes=num_keypoint_classes,
                    max_num_keypoints=max_num_keypoints,
                )

            results.append(
                {
                    "scores": scores_i,
                    "labels": labels_i,
                    "boxes": boxes_i,
                    "keypoints": output_keypoints,
                    "keypoint_precision_cholesky": output_keypoint_precision,
                }
            )
        return results

    @staticmethod
    def _gather_keypoints_for_queries(out_keypoints_i: torch.Tensor, query_indices: torch.Tensor) -> torch.Tensor:
        """Gather keypoint predictions for the selected query rows of one image.

        Args:
            out_keypoints_i: Keypoint predictions for one image with shape
                ``(Q, C * max(K_c), D)``.
            query_indices: Top-k query indices for that image.

        Returns:
            Keypoint predictions aligned with the selected detections.
        """
        return torch.gather(
            out_keypoints_i,
            0,
            query_indices.unsqueeze(-1).unsqueeze(-1).repeat(1, out_keypoints_i.shape[-2], out_keypoints_i.shape[-1]),
        )

    @staticmethod
    def _empty_keypoint_outputs(
        keypoints_i: torch.Tensor,
        max_num_keypoints: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create zero/NaN-filled keypoint output tensors for one image.

        Args:
            keypoints_i: Gathered selected keypoint predictions for one image.
            max_num_keypoints: Maximum active keypoint count across classes.

        Returns:
            Tuple of pixel-space keypoint output ``(x, y, confidence)`` and raw
            precision-Cholesky output ``(log_l11, l21, log_l22)``. Inactive
            class-padded keypoints remain zeros for coordinates/confidence and
            NaN for precision.
        """
        output_keypoints = keypoints_i.new_zeros((keypoints_i.shape[0], max_num_keypoints, 3))
        output_keypoint_precision = keypoints_i.new_full((keypoints_i.shape[0], max_num_keypoints, 3), float("nan"))
        return output_keypoints, output_keypoint_precision

    def _decode_keypoints_for_image(
        self,
        *,
        keypoints_i: torch.Tensor,
        labels_i: torch.Tensor,
        scores_i: torch.Tensor,
        target_size: torch.Tensor,
        output_keypoints: torch.Tensor,
        output_keypoint_precision: torch.Tensor,
        num_keypoint_classes: int,
        max_num_keypoints: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode selected keypoints for one image into output tensors.

        Args:
            keypoints_i: Gathered selected keypoint predictions for one image.
            labels_i: Predicted class labels for selected detections.
            scores_i: Object scores before optional keypoint uncertainty fusion.
            target_size: Image ``(height, width)`` tensor.
            output_keypoints: Preallocated keypoint output tensor to fill.
            output_keypoint_precision: Preallocated precision output tensor to
                fill.
            num_keypoint_classes: Number of class slots in the keypoint schema.
            max_num_keypoints: Padded keypoint count per class slot.

        Returns:
            Updated object scores, keypoint outputs, and precision outputs. The
            score tensor is cloned only when uncertainty fusion is applied.

        Raises:
            ValueError: If the padded keypoint dimension of ``keypoints_i`` is
                not equal to ``num_keypoint_classes * max_num_keypoints``.
        """
        total_padded_keypoint_slots = keypoints_i.shape[1]
        if total_padded_keypoint_slots != num_keypoint_classes * max_num_keypoints:
            raise ValueError(
                f"keypoints_i padded slot dimension ({total_padded_keypoint_slots}) must equal "
                f"num_keypoint_classes ({num_keypoint_classes}) * max_num_keypoints ({max_num_keypoints})."
            )
        reshaped = keypoints_i.view(
            keypoints_i.shape[0], num_keypoint_classes, max_num_keypoints, keypoints_i.shape[-1]
        )
        valid_class_mask = labels_i < num_keypoint_classes
        if not valid_class_mask.any():
            return scores_i, output_keypoints, output_keypoint_precision

        valid_indices = valid_class_mask.nonzero(as_tuple=True)[0]
        selected_labels = labels_i[valid_indices]
        selected_keypoints = reshaped[valid_indices, selected_labels]
        if self.trace_alpha > 0 and selected_keypoints.shape[-1] >= 7:
            scores_i = self._apply_keypoint_trace_fusion(scores_i, valid_indices, selected_labels, selected_keypoints)

        img_h, img_w = target_size
        has_precision = selected_keypoints.shape[-1] >= 7
        # Iterate over the (small) set of keypoint classes rather than per detection.
        # Avoids `num_select` GPU->CPU stream syncs from `.item()`; loop bound is
        # `num_keypoint_classes` (typically 1-20) instead of `num_select` (up to 300).
        for class_idx in range(num_keypoint_classes):
            class_mask = selected_labels == class_idx
            if not class_mask.any():
                continue
            num_active_keypoints = self.num_keypoints_per_class[class_idx]
            if num_active_keypoints <= 0:
                continue

            out_idx = valid_indices[class_mask]
            active_keypoints = selected_keypoints[class_mask, :num_active_keypoints]
            output_keypoints[out_idx, :num_active_keypoints, 0] = active_keypoints[..., 0] * img_w
            output_keypoints[out_idx, :num_active_keypoints, 1] = active_keypoints[..., 1] * img_h
            output_keypoints[out_idx, :num_active_keypoints, 2] = active_keypoints[..., 2].sigmoid()
            if has_precision:
                output_keypoint_precision[out_idx, :num_active_keypoints] = active_keypoints[..., 4:7]

        return scores_i, output_keypoints, output_keypoint_precision

    def _apply_keypoint_trace_fusion(
        self,
        scores_i: torch.Tensor,
        valid_indices: torch.Tensor,
        selected_labels: torch.Tensor,
        selected_keypoints: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse object scores with keypoint localization uncertainty.

        Args:
            scores_i: Object scores for one image before keypoint uncertainty
                fusion.
            valid_indices: Detection indices whose predicted class has a
                keypoint schema slot.
            selected_labels: Class labels corresponding to ``valid_indices``.
            selected_keypoints: Class-selected keypoint predictions for those
                detections.

        Returns:
            A score tensor where valid keypoint detections are multiplied by
            ``exp(-trace_alpha * log_mean_trace)`` and then normalized to
            ``[0, 1)``. The trace is the findability-weighted mean expected
            squared localization error implied by the predicted
            precision-Cholesky parameters.
        """
        num_keypoint_classes = len(self.num_keypoints_per_class)
        log_mean_traces = selected_keypoints.new_zeros(selected_labels.shape[0])
        # Iterate over the small set of classes rather than per detection — the
        # per-detection loop required `.item()` on every iteration (GPU->CPU
        # stream sync) and a final `torch.stack` over a Python list. Grouping by
        # class lets us call `_keypoint_log_mean_trace` once on a batched tensor
        # whose `num_active_keypoints` is constant within the class.
        for class_idx in range(num_keypoint_classes):
            class_mask = selected_labels == class_idx
            if not class_mask.any():
                continue
            num_active_keypoints = self.num_keypoints_per_class[class_idx]
            if num_active_keypoints <= 0:
                # Defaults to 0.0 from `new_zeros` above — matches the legacy
                # per-detection branch that appended a 0 tensor.
                continue

            active_keypoints = selected_keypoints[class_mask, :num_active_keypoints]
            log_mean_traces[class_mask] = self._keypoint_log_mean_trace(active_keypoints)

        scores_i = scores_i.clone()
        log_fused_scores = torch.log(scores_i[valid_indices]) - self.trace_alpha * log_mean_traces
        normalized_scores = torch.sigmoid(log_fused_scores)
        # Keep the public contract strictly below 1.0 even when the log-fused score saturates.
        max_score = torch.nextafter(
            torch.ones_like(normalized_scores),
            torch.zeros_like(normalized_scores),
        )
        scores_i[valid_indices] = torch.minimum(normalized_scores, max_score)
        return scores_i

    @staticmethod
    def _keypoint_log_mean_trace(active_keypoints: torch.Tensor) -> torch.Tensor:
        """Compute log mean covariance trace for active keypoints.

        Args:
            active_keypoints: Active keypoint predictions with shape
                ``(..., K, D)`` where ``K`` is the active keypoint count for the
                detection's class and ``D`` is the per-keypoint feature dim.
                Columns ``4:7`` are ``(log_l11, l21, log_l22)`` precision
                Cholesky parameters and column ``2`` is the findable logit. A
                leading batch dimension is supported so this can be called once
                for an entire class group of detections.

        Returns:
            Tensor with the leading batch dimensions of ``active_keypoints``
            (scalar when called on a single detection). Each entry is the log of
            the findability-weighted arithmetic mean trace of the covariance
            matrix. The computation stays in log space for numerical stability
            with very sharp or very uncertain predictions.
        """
        log_l11 = active_keypoints[..., 4]
        l21 = active_keypoints[..., 5]
        log_l22 = active_keypoints[..., 6]
        w_find = active_keypoints[..., 2].sigmoid()
        log_t1 = -2.0 * log_l11
        log_t2 = -2.0 * log_l22
        log_t3 = 2.0 * torch.log(l21.abs().clamp(min=1e-12)) + log_t1 + log_t2
        log_trace_sigma = torch.logsumexp(torch.stack([log_t1, log_t2, log_t3], dim=-1), dim=-1)
        log_w_find = torch.log(w_find.clamp(min=1e-12))
        return torch.logsumexp(log_trace_sigma + log_w_find, dim=-1) - torch.logsumexp(log_w_find, dim=-1)

    @staticmethod
    def _postprocess_boxes(
        scores: torch.Tensor,
        labels: torch.Tensor,
        boxes: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Build detection-only result dictionaries.

        Args:
            scores: Selected object scores with shape ``(B, K)``.
            labels: Selected class labels with shape ``(B, K)``.
            boxes: Selected absolute boxes with shape ``(B, K, 4)``.

        Returns:
            One result dict per image containing scores, labels, and boxes.
        """
        return [{"scores": score, "labels": label, "boxes": box} for score, label, box in zip(scores, labels, boxes)]

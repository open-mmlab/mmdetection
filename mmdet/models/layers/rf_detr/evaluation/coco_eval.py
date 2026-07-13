# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""COCO evaluator for ONNX/TRT export benchmarking.

Provides :class:`CocoEvaluator` used by :mod:`rfdetr.export.benchmark` to compute mAP during ONNX and TensorRT inference
benchmarks.

Implementation mirrors torchvision's evaluator structure but uses ``faster_coco_eval`` as the runtime backend.
"""

from __future__ import annotations

import contextlib
import copy
import os
from dataclasses import dataclass
from typing import Any, cast

import faster_coco_eval.core.mask as mask_util
import numpy as np
from faster_coco_eval import COCO
from faster_coco_eval.core.faster_eval_api import COCOeval

from mmdet.models.layers.rf_detr.utilities.distributed import all_gather
from mmdet.models.layers.rf_detr.utilities.logger import get_logger

logger = get_logger()

_COCO_PERSON_KEYPOINT_SIGMAS = (
    np.asarray(
        [
            0.26,
            0.25,
            0.25,
            0.35,
            0.35,
            0.79,
            0.79,
            0.72,
            0.72,
            0.62,
            0.62,
            1.07,
            1.07,
            0.87,
            0.87,
            0.89,
            0.89,
        ],
        dtype=np.float32,
    )
    / 10.0
)
_DEFAULT_CUSTOM_KEYPOINT_OKS_SIGMA = 0.05
_WARNED_CUSTOM_KEYPOINT_OKS_COUNTS: set[int] = set()


@dataclass(frozen=True, slots=True)
class _KeypointCategoryGroup:
    """Keypoint categories sharing one keypoint count and OKS sigma vector."""

    category_ids: list[int]
    keypoint_count: int
    keypoint_oks_sigmas: list[float] | None


@dataclass(slots=True)
class _GroupedKeypointCOCOeval:
    """Aggregate COCO keypoint stats for categories with different keypoint counts."""

    groups: list[_KeypointCategoryGroup]
    stats: np.ndarray[Any, Any]
    evals: list[COCOeval]


def _ensure_faster_coco(coco_gt: Any) -> COCO:
    """Return a faster-coco-eval COCO object for evaluator construction."""
    if isinstance(coco_gt, COCO) and hasattr(coco_gt, "cat_img_map"):
        return coco_gt

    faster_coco = COCO()
    faster_coco.dataset = copy.deepcopy(coco_gt.dataset)
    faster_coco.createIndex()
    label2cat = getattr(coco_gt, "label2cat", None)
    if label2cat is not None:
        faster_coco.label2cat = copy.deepcopy(label2cat)
    return faster_coco


def _backfill_num_keypoints(coco_gt: COCO) -> None:
    """Populate missing COCO ``num_keypoints`` fields from visibility flags."""
    annotations_by_id: dict[int, dict[str, Any]] = {}
    for annotation in getattr(coco_gt, "dataset", {}).get("annotations", []):
        annotation_id = annotation.get("id")
        if isinstance(annotation_id, int):
            annotations_by_id[annotation_id] = annotation
        keypoints = annotation.get("keypoints")
        if "num_keypoints" not in annotation and isinstance(keypoints, list):
            annotation["num_keypoints"] = sum(1 for visibility in keypoints[2::3] if visibility > 0)

    for annotation_id, annotation in getattr(coco_gt, "anns", {}).items():
        keypoints = annotation.get("keypoints")
        if "num_keypoints" not in annotation and isinstance(keypoints, list):
            annotation["num_keypoints"] = sum(1 for visibility in keypoints[2::3] if visibility > 0)
        dataset_annotation = annotations_by_id.get(annotation_id)
        if dataset_annotation is not None and "num_keypoints" in annotation:
            dataset_annotation["num_keypoints"] = annotation["num_keypoints"]


def _infer_keypoint_count(coco_gt: COCO) -> int | None:
    """Infer a single keypoint count from COCO category metadata or annotations."""
    counts = {count for count in _infer_keypoint_counts_by_category(coco_gt).values() if count > 0}

    if not counts:
        return None
    if len(counts) > 1:
        raise ValueError(
            "COCO keypoint evaluation requires one keypoint count across evaluated categories; "
            f"found counts {sorted(counts)}."
        )
    return next(iter(counts))


def _infer_keypoint_counts_by_category(coco_gt: COCO) -> dict[int, int]:
    """Infer keypoint count for each category from COCO category metadata or annotations."""
    counts: dict[int, int] = {}
    for category_id, category in getattr(coco_gt, "cats", {}).items():
        keypoints = category.get("keypoints")
        if isinstance(keypoints, list) and keypoints:
            counts[int(category_id)] = len(keypoints)

    for annotation in getattr(coco_gt, "dataset", {}).get("annotations", []):
        keypoints = annotation.get("keypoints")
        if not isinstance(keypoints, list) or not keypoints:
            continue
        category_id = int(annotation["category_id"])
        counts[category_id] = max(counts.get(category_id, 0), len(keypoints) // 3)
    for annotation in getattr(coco_gt, "anns", {}).values():
        keypoints = annotation.get("keypoints")
        if not isinstance(keypoints, list) or not keypoints:
            continue
        category_id = int(annotation["category_id"])
        counts[category_id] = max(counts.get(category_id, 0), len(keypoints) // 3)
    return counts


def _resolve_keypoint_oks_sigmas(coco_gt: COCO, keypoint_oks_sigmas: list[float] | None) -> list[float] | None:
    """Resolve OKS sigmas for faster-coco-eval keypoint evaluation."""
    keypoint_count = _infer_keypoint_count(coco_gt)
    if keypoint_oks_sigmas is not None:
        sigmas = np.asarray(keypoint_oks_sigmas, dtype=np.float32)
        if sigmas.ndim != 1 or sigmas.size == 0:
            raise ValueError("keypoint_oks_sigmas must be a non-empty one-dimensional sequence.")
        if not np.isfinite(sigmas).all() or np.any(sigmas <= 0):
            raise ValueError("keypoint_oks_sigmas values must be positive finite numbers.")
        if keypoint_count is not None and sigmas.size != keypoint_count:
            raise ValueError(
                f"keypoint_oks_sigmas length {sigmas.size} does not match dataset keypoint count {keypoint_count}."
            )
        return cast(list[float], sigmas.tolist())

    if keypoint_count is None or keypoint_count == len(_COCO_PERSON_KEYPOINT_SIGMAS):
        return None

    _warn_custom_keypoint_oks_sigma_once(keypoint_count)
    return np.full(keypoint_count, _DEFAULT_CUSTOM_KEYPOINT_OKS_SIGMA, dtype=np.float32).tolist()


def _resolve_group_keypoint_oks_sigmas(
    keypoint_count: int,
    keypoint_oks_sigmas: list[float] | None,
) -> list[float] | None:
    """Resolve OKS sigmas for one keypoint-count group."""
    if keypoint_oks_sigmas is None:
        if keypoint_count == len(_COCO_PERSON_KEYPOINT_SIGMAS):
            return None
        _warn_custom_keypoint_oks_sigma_once(keypoint_count)
        return np.full(keypoint_count, _DEFAULT_CUSTOM_KEYPOINT_OKS_SIGMA, dtype=np.float32).tolist()

    sigmas = np.asarray(keypoint_oks_sigmas, dtype=np.float32)
    if sigmas.ndim != 1 or sigmas.size == 0:
        raise ValueError("keypoint_oks_sigmas must be a non-empty one-dimensional sequence.")
    if not np.isfinite(sigmas).all() or np.any(sigmas <= 0):
        raise ValueError("keypoint_oks_sigmas values must be positive finite numbers.")
    if sigmas.size < keypoint_count:
        raise ValueError(
            f"keypoint_oks_sigmas length {sigmas.size} does not match dataset keypoint count {keypoint_count}."
        )
    return cast(list[float], sigmas[:keypoint_count].tolist())


def _warn_custom_keypoint_oks_sigma_once(keypoint_count: int) -> None:
    """Warn once per keypoint count when using uniform custom OKS sigmas."""
    if keypoint_count in _WARNED_CUSTOM_KEYPOINT_OKS_COUNTS:
        return
    _WARNED_CUSTOM_KEYPOINT_OKS_COUNTS.add(keypoint_count)
    logger.warning(
        "COCO keypoint metadata defines %s keypoints, but no keypoint_oks_sigmas were provided. "
        "Using uniform OKS sigma %.3f for custom keypoint evaluation.",
        keypoint_count,
        _DEFAULT_CUSTOM_KEYPOINT_OKS_SIGMA,
    )


def _build_keypoint_category_groups(
    coco_gt: COCO,
    keypoint_oks_sigmas: list[float] | None,
) -> list[_KeypointCategoryGroup]:
    """Build category groups that can each be evaluated by one COCO keypoint evaluator."""
    counts_by_category = _infer_keypoint_counts_by_category(coco_gt)
    grouped_category_ids: dict[int, list[int]] = {}
    for category_id, keypoint_count in counts_by_category.items():
        if keypoint_count <= 0:
            continue
        grouped_category_ids.setdefault(keypoint_count, []).append(category_id)
    return [
        _KeypointCategoryGroup(
            category_ids=sorted(category_ids),
            keypoint_count=keypoint_count,
            keypoint_oks_sigmas=_resolve_group_keypoint_oks_sigmas(keypoint_count, keypoint_oks_sigmas),
        )
        for keypoint_count, category_ids in sorted(grouped_category_ids.items())
    ]


def _filter_coco_by_category_ids(coco_gt: COCO, category_ids: list[int]) -> COCO:
    """Return a COCO object containing only the requested categories and annotations."""
    category_id_set = set(category_ids)
    gt_dataset: dict[str, Any] = copy.deepcopy(coco_gt.dataset)
    dataset: dict[str, Any] = {
        **gt_dataset,
        "categories": [
            category for category in gt_dataset.get("categories", []) if int(category["id"]) in category_id_set
        ],
        "annotations": [
            annotation
            for annotation in gt_dataset.get("annotations", [])
            if int(annotation.get("category_id", -1)) in category_id_set
        ],
    }

    filtered = COCO()
    filtered.dataset = dataset
    filtered.createIndex()
    label2cat = getattr(coco_gt, "label2cat", None)
    if label2cat is not None:
        filtered.label2cat = {label: cat_id for label, cat_id in label2cat.items() if cat_id in category_id_set}
    return filtered


def _load_coco_results(coco_gt: COCO, results: list[dict[str, Any]]) -> COCO:
    """Build a COCO detections object, including the empty-result case."""
    if results:
        return coco_gt.loadRes(results)

    coco_dt = COCO()
    gt_dataset: dict[str, Any] = coco_gt.dataset
    coco_dt.dataset = {
        "info": copy.deepcopy(gt_dataset.get("info", {})),
        "images": copy.deepcopy(gt_dataset.get("images", [])),
        "categories": copy.deepcopy(gt_dataset.get("categories", [])),
        "annotations": [],
    }
    coco_dt.createIndex()
    return coco_dt


def _xyxy_to_xywh(boxes: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Convert boxes from [x1, y1, x2, y2] to [x1, y1, w, h]."""
    boxes = boxes.copy()
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]
    return boxes


def _weighted_mean_coco_stats(stats: list[np.ndarray[Any, Any]], weights: list[int]) -> np.ndarray[Any, Any]:
    """Compute category-weighted mean COCO stats, ignoring unavailable ``-1`` values."""
    if not stats:
        return np.full((10,), -1.0, dtype=np.float32)

    max_len = max(len(item) for item in stats)
    aggregated = np.full((max_len,), -1.0, dtype=np.float32)
    for stat_idx in range(max_len):
        numerator = 0.0
        denominator = 0
        for stat, weight in zip(stats, weights):
            if len(stat) <= stat_idx or stat[stat_idx] < 0:
                continue
            numerator += float(stat[stat_idx]) * weight
            denominator += weight
        if denominator > 0:
            aggregated[stat_idx] = numerator / denominator
    return aggregated


def _log_keypoint_stats(stats: np.ndarray[Any, Any]) -> None:
    """Log keypoint COCO stats from an already accumulated evaluator."""
    labels = (
        ("Average Precision", "(AP)", "0.50:0.95", "all", 20, 0),
        ("Average Precision", "(AP)", "0.50", "all", 20, 1),
        ("Average Precision", "(AP)", "0.75", "all", 20, 2),
        ("Average Precision", "(AP)", "0.50:0.95", "medium", 20, 3),
        ("Average Precision", "(AP)", "0.50:0.95", "large", 20, 4),
        ("Average Recall", "(AR)", "0.50:0.95", "all", 20, 5),
        ("Average Recall", "(AR)", "0.50", "all", 20, 6),
        ("Average Recall", "(AR)", "0.75", "all", 20, 7),
        ("Average Recall", "(AR)", "0.50:0.95", "medium", 20, 8),
        ("Average Recall", "(AR)", "0.50:0.95", "large", 20, 9),
    )
    log_template = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
    for title, metric_type, iou, area, max_dets, stat_idx in labels:
        value = float(stats[stat_idx]) if len(stats) > stat_idx else -1.0
        logger.info(log_template.format(title, metric_type, iou, area, max_dets, value))


def _accumulate_and_summarize(coco_eval: COCOeval, *, log_summary: bool) -> None:
    """Accumulate a COCO evaluator and populate ``stats`` regardless of log mode."""
    if log_summary:
        coco_eval.accumulate()
        patched_pycocotools_summarize(coco_eval)
        return

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        coco_eval.accumulate()
        patched_pycocotools_summarize(coco_eval, log_summary=False)


class CocoEvaluator:
    """COCO evaluator that works in distributed mode."""

    def __init__(
        self,
        coco_gt: Any,
        iou_types: list[str],
        max_dets: int = 100,
        keypoint_oks_sigmas: list[float] | None = None,
        log_summary: bool = True,
    ) -> None:
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(_ensure_faster_coco(coco_gt))
        resolved_keypoint_oks_sigmas = None
        keypoint_category_groups: list[_KeypointCategoryGroup] = []
        if "keypoints" in iou_types:
            _backfill_num_keypoints(coco_gt)
            keypoint_category_groups = _build_keypoint_category_groups(coco_gt, keypoint_oks_sigmas)
            if len(keypoint_category_groups) <= 1:
                resolved_keypoint_oks_sigmas = _resolve_keypoint_oks_sigmas(coco_gt, keypoint_oks_sigmas)
        self.coco_gt = coco_gt
        self.max_dets = max_dets
        # label2cat maps contiguous model label indices back to original COCO category_ids.
        # Set by CocoDetection when cat2label remapping is active; None otherwise.
        self.label2cat: dict[int, int] | None = getattr(coco_gt, "label2cat", None)

        self.iou_types = iou_types
        self.coco_eval: dict[str, COCOeval | _GroupedKeypointCOCOeval] = {}
        for iou_type in iou_types:
            if iou_type == "keypoints" and len(keypoint_category_groups) > 1:
                self.coco_eval[iou_type] = _GroupedKeypointCOCOeval(
                    groups=keypoint_category_groups,
                    stats=np.full((10,), -1.0, dtype=np.float32),
                    evals=[],
                )
                continue
            kwargs = {"kpt_oks_sigmas": resolved_keypoint_oks_sigmas} if iou_type == "keypoints" else {}
            coco_eval = COCOeval(coco_gt, iouType=iou_type, **kwargs)
            coco_eval.params.maxDets = [20] if iou_type == "keypoints" else [1, 10, max_dets]
            self.coco_eval[iou_type] = coco_eval

        self.img_ids: list[int] = []
        self.coco_results: dict[str, list[dict[str, Any]]] = {k: [] for k in iou_types}
        self.cat_ids = set(coco_gt.cats.keys())
        self._keypoint_counts_by_category = _infer_keypoint_counts_by_category(coco_gt)
        self._prefer_raw_category_ids = False
        self._log_summary = log_summary

    def _resolve_category_id(self, label: int, use_raw_category_ids: bool) -> int | None:
        """Resolve a predicted label to a COCO category_id."""
        if use_raw_category_ids:
            return label if label in self.cat_ids else None
        if self.label2cat is not None:
            category_id = self.label2cat.get(label)
            return category_id if category_id in self.cat_ids else None
        if label in self.cat_ids:
            return label
        return None

    def _should_use_raw_category_ids(self, labels: list[int]) -> bool:
        """Detect whether model predictions are already raw COCO category IDs."""
        if self.label2cat is None:
            return True
        if self._prefer_raw_category_ids:
            return True
        uses_raw_ids = list(self.label2cat.keys()) == list(self.label2cat.values())
        if uses_raw_ids:
            self._prefer_raw_category_ids = True
            return True
        return False

    def update(self, predictions: dict[int, Any]) -> None:
        """Accumulate per-image predictions."""
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            self.coco_results[iou_type].extend(results)

    def synchronize_between_processes(self) -> None:
        """Merge image IDs and COCO result records across distributed processes.

        Each image ID is assigned to exactly one rank (first rank that reports it), so
        predictions for images that appear on multiple ranks due to
        ``DistributedSampler(drop_last=False)`` padding are included only once.  Without
        this deduplication, padded images produce duplicate DT entries that compete for
        the per-image ``maxDets`` cap and bias mAP upward in early epochs then downward
        as genuine new detections are displaced — the "peak-then-decrease" pattern.

        Contract:
            This method assumes that when the same ``image_id`` appears on multiple ranks
            its predictions are *identical* (DDP-padding duplicates).  It is **not** safe
            for evaluation strategies where independent predictions for the same image are
            produced on different ranks, because only the first rank's predictions are kept
            and the rest are silently discarded.

        Single-process path:
            When ``all_gather`` is called in a non-distributed context it returns a
            single-element list ``[x]``, so ``rank_idx`` is always 0 and every result
            passes the ownership check unchanged — the dedup loop is a no-op and all
            results are preserved.  This makes the method safe for single-GPU and ONNX/TRT
            benchmark use via :mod:`rfdetr.export.benchmark`.
        """
        gathered_img_ids = all_gather(self.img_ids)
        # First rank to report an image_id owns it; all other ranks' predictions for
        # that image_id are dropped (they are identical copies from DDP padding).
        img_id_to_rank: dict[int, int] = {}
        for rank_idx, rank_img_ids in enumerate(gathered_img_ids):
            for img_id in rank_img_ids:
                if img_id not in img_id_to_rank:
                    img_id_to_rank[img_id] = rank_idx
        self.img_ids = sorted(img_id_to_rank.keys())
        for iou_type in self.iou_types:
            gathered_results = all_gather(self.coco_results[iou_type])
            deduped: list[dict[str, Any]] = []
            for rank_idx, rank_results in enumerate(gathered_results):
                for result in rank_results:
                    if img_id_to_rank.get(result["image_id"]) == rank_idx:
                        deduped.append(result)
            self.coco_results[iou_type] = deduped

    def accumulate(self) -> None:
        """Accumulate per-image evaluation results into mean metrics."""
        for iou_type, coco_eval in self.coco_eval.items():
            if isinstance(coco_eval, _GroupedKeypointCOCOeval):
                self._evaluate_grouped_keypoints(coco_eval)
                if self._log_summary:
                    _log_keypoint_stats(coco_eval.stats)
                continue
            self._evaluate(iou_type, coco_eval)
            _accumulate_and_summarize(coco_eval, log_summary=self._log_summary)

    def summarize(self) -> None:
        """Print and log COCO summary statistics."""
        for iou_type, coco_eval in self.coco_eval.items():
            logger.info(f"IoU metric: {iou_type}")
            if isinstance(coco_eval, _GroupedKeypointCOCOeval):
                _log_keypoint_stats(coco_eval.stats)
            else:
                patched_pycocotools_summarize(coco_eval)

    def _evaluate(self, iou_type: str, coco_eval: COCOeval) -> None:
        """Run faster-coco-eval evaluation for accumulated COCO result records."""
        results = self.coco_results[iou_type]
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            coco_dt = _load_coco_results(self.coco_gt, results)
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(np.unique(self.img_ids))
            coco_eval.evaluate()

    def _evaluate_grouped_keypoints(self, grouped_eval: _GroupedKeypointCOCOeval) -> None:
        """Run keypoint evaluation per keypoint-count group and aggregate stats."""
        grouped_eval.evals = []
        group_stats: list[np.ndarray[Any, Any]] = []
        group_weights: list[int] = []
        all_results = self.coco_results["keypoints"]
        img_ids = list(np.unique(self.img_ids))

        for group in grouped_eval.groups:
            category_id_set = set(group.category_ids)
            group_gt = _filter_coco_by_category_ids(self.coco_gt, group.category_ids)
            group_results = [result for result in all_results if int(result["category_id"]) in category_id_set]
            group_coco_eval = COCOeval(
                group_gt,
                iouType="keypoints",
                kpt_oks_sigmas=group.keypoint_oks_sigmas,
            )
            group_coco_eval.params.maxDets = [20]
            self._evaluate_grouped_keypoint_results(group_coco_eval, group_gt, group_results, img_ids)
            _accumulate_and_summarize(group_coco_eval, log_summary=False)
            grouped_eval.evals.append(group_coco_eval)
            group_stats.append(np.asarray(getattr(group_coco_eval, "stats"), dtype=np.float32))
            group_weights.append(len(group.category_ids))

        grouped_eval.stats = _weighted_mean_coco_stats(group_stats, group_weights)

    @staticmethod
    def _evaluate_grouped_keypoint_results(
        coco_eval: COCOeval,
        coco_gt: COCO,
        results: list[dict[str, Any]],
        img_ids: list[int],
    ) -> None:
        """Evaluate one grouped keypoint result set."""
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            coco_dt = _load_coco_results(coco_gt, results)
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = img_ids
            coco_eval.evaluate()

    def prepare(self, predictions: dict[int, Any], iou_type: str) -> list[dict[str, Any]]:
        """Convert predictions to COCO format for the given iou_type."""
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions: dict[int, Any]) -> list[dict[str, Any]]:
        """Format bounding-box predictions as COCO result dicts."""
        coco_results: list[dict[str, Any]] = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = _xyxy_to_xywh(boxes.cpu().numpy()).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            use_raw_category_ids = self._should_use_raw_category_ids(labels)
            for k, box in enumerate(boxes):
                category_id = self._resolve_category_id(labels[k], use_raw_category_ids)
                if category_id is None:
                    continue
                coco_results.append(
                    {
                        "image_id": original_id,
                        "category_id": category_id,
                        "bbox": box,
                        "score": scores[k],
                    }
                )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions: dict[int, Any]) -> list[dict[str, Any]]:
        """Format segmentation mask predictions as COCO result dicts."""
        coco_results: list[dict[str, Any]] = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            use_raw_category_ids = self._should_use_raw_category_ids(labels)

            encode = getattr(mask_util, "encode")
            rles: list[dict[str, Any]] = [
                encode(np.array(mask.cpu()[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0] for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            for k, rle in enumerate(rles):
                category_id = self._resolve_category_id(labels[k], use_raw_category_ids)
                if category_id is None:
                    continue
                coco_results.append(
                    {
                        "image_id": original_id,
                        "category_id": category_id,
                        "segmentation": rle,
                        "score": scores[k],
                    }
                )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions: dict[int, Any]) -> list[dict[str, Any]]:
        """Format keypoint predictions as COCO result dicts."""
        coco_results: list[dict[str, Any]] = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = _xyxy_to_xywh(boxes.cpu().numpy()).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()
            use_raw_category_ids = self._should_use_raw_category_ids(labels)
            for k, keypoint in enumerate(keypoints):
                category_id = self._resolve_category_id(labels[k], use_raw_category_ids)
                if category_id is None:
                    continue
                keypoint_count = self._keypoint_counts_by_category.get(category_id)
                if keypoint_count is not None:
                    keypoint = keypoint[: keypoint_count * 3]
                coco_results.append(
                    {
                        "image_id": original_id,
                        "category_id": category_id,
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                )
        return coco_results


#################################################################
# From pycocotools, patched first _summarize() call to use
# maxDets[-1] instead of hardcoded 100.
#################################################################
def patched_pycocotools_summarize(self: COCOeval, *, log_summary: bool = True) -> None:
    """Compute and display summary metrics for evaluation results.

    Args:
        self: COCOeval instance that has already run ``accumulate()``.
        log_summary: Whether to log per-metric lines at INFO level.

    Raises:
        Exception: If ``accumulate()`` has not been called (``self.eval`` is empty).
        ValueError: If ``self.params.iouType`` is not ``"segm"``, ``"bbox"``, or
            ``"keypoints"``.
    """
    p = self.params
    evaluation = self.eval

    def _summarize(ap: int = 1, iou_thr: float | None = None, area_rng: str = "all", max_dets: int = 100) -> float:
        log_template = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
        title_str = "Average Precision" if ap == 1 else "Average Recall"
        type_str = "(AP)" if ap == 1 else "(AR)"
        iou_str = f"{p.iouThrs[0]:0.2f}:{p.iouThrs[-1]:0.2f}" if iou_thr is None else f"{iou_thr:0.2f}"

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == area_rng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == max_dets]
        if ap == 1:
            s = evaluation["precision"]
            if iou_thr is not None:
                t = np.where(iou_thr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        else:
            s = evaluation["recall"]
            if iou_thr is not None:
                t = np.where(iou_thr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        mean_s = -1 if len(s[s > -1]) == 0 else float(cast(Any, np.mean(s[s > -1])))
        if log_summary:
            logger.info(log_template.format(title_str, type_str, iou_str, area_rng, max_dets, mean_s))
        return mean_s

    def _summarizeDets() -> np.ndarray[Any, Any]:  # noqa: N802
        stats = np.zeros((12,))
        stats[0] = _summarize(1, max_dets=p.maxDets[2])
        stats[1] = _summarize(1, iou_thr=0.5, max_dets=p.maxDets[2])
        stats[2] = _summarize(1, iou_thr=0.75, max_dets=p.maxDets[2])
        stats[3] = _summarize(1, area_rng="small", max_dets=p.maxDets[2])
        stats[4] = _summarize(1, area_rng="medium", max_dets=p.maxDets[2])
        stats[5] = _summarize(1, area_rng="large", max_dets=p.maxDets[2])
        stats[6] = _summarize(0, max_dets=p.maxDets[0])
        stats[7] = _summarize(0, max_dets=p.maxDets[1])
        stats[8] = _summarize(0, max_dets=p.maxDets[2])
        stats[9] = _summarize(0, area_rng="small", max_dets=p.maxDets[2])
        stats[10] = _summarize(0, area_rng="medium", max_dets=p.maxDets[2])
        stats[11] = _summarize(0, area_rng="large", max_dets=p.maxDets[2])
        return stats

    def _summarizeKps() -> np.ndarray[Any, Any]:  # noqa: N802
        stats = np.zeros((10,))
        stats[0] = _summarize(1, max_dets=20)
        stats[1] = _summarize(1, max_dets=20, iou_thr=0.5)
        stats[2] = _summarize(1, max_dets=20, iou_thr=0.75)
        stats[3] = _summarize(1, max_dets=20, area_rng="medium")
        stats[4] = _summarize(1, max_dets=20, area_rng="large")
        stats[5] = _summarize(0, max_dets=20)
        stats[6] = _summarize(0, max_dets=20, iou_thr=0.5)
        stats[7] = _summarize(0, max_dets=20, iou_thr=0.75)
        stats[8] = _summarize(0, max_dets=20, area_rng="medium")
        stats[9] = _summarize(0, max_dets=20, area_rng="large")
        return stats

    if not evaluation:
        raise Exception("Please run accumulate() first")
    iou_type = p.iouType
    if iou_type == "segm" or iou_type == "bbox":
        summarize = _summarizeDets
    elif iou_type == "keypoints":
        summarize = _summarizeKps
    else:
        raise ValueError(f"Unknown iou type {iou_type}")
    self.stats = summarize()

# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Confidence-threshold sweep for precision/recall/F1 computation."""

from __future__ import annotations

from typing import Any

import numpy as np


def sweep_confidence_thresholds(
    per_class_data: list[dict[str, Any]],
    conf_thresholds: Any,
    classes_with_gt: list[int],
) -> list[dict[str, Any]]:
    """Sweep confidence thresholds and compute precision/recall/F1 at each.

    Args:
        per_class_data: Per-class matching data list indexed by class id.
            Each entry is a dict with keys ``"scores"``, ``"matches"``, ``"ignore"``, and ``"total_gt"``.
        conf_thresholds: Iterable of float confidence thresholds to evaluate.
        classes_with_gt: List of class indices that have at least one GT instance — used for macro-averaging.

    Returns:
        List of result dicts, one per threshold, each containing:
            - ``"confidence_threshold"``: float
            - ``"macro_f1"``: float
            - ``"macro_precision"``: float
            - ``"macro_recall"``: float
            - ``"per_class_prec"``: float ndarray
            - ``"per_class_rec"``: float ndarray
            - ``"per_class_f1"``: float ndarray
    """
    num_classes = len(per_class_data)
    results: list[dict[str, Any]] = []

    for conf_thresh in conf_thresholds:
        per_class_precisions: list[float] = []
        per_class_recalls: list[float] = []
        per_class_f1s: list[float] = []

        for k in range(num_classes):
            data = per_class_data[k]
            scores = data["scores"]
            matches = data["matches"]
            ignore = data["ignore"]
            total_gt = data["total_gt"]

            above_thresh = scores >= conf_thresh
            valid = above_thresh & ~ignore

            valid_matches = matches[valid]

            tp = np.sum(valid_matches != 0)
            fp = np.sum(valid_matches == 0)
            fn = total_gt - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class_precisions.append(precision)
            per_class_recalls.append(recall)
            per_class_f1s.append(f1)

        if len(classes_with_gt) > 0:
            macro_precision = float(np.mean([per_class_precisions[k] for k in classes_with_gt]))
            macro_recall = float(np.mean([per_class_recalls[k] for k in classes_with_gt]))
            macro_f1 = float(np.mean([per_class_f1s[k] for k in classes_with_gt]))
        else:
            macro_precision = 0.0
            macro_recall = 0.0
            macro_f1 = 0.0

        results.append(
            {
                "confidence_threshold": conf_thresh,
                "macro_f1": macro_f1,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "per_class_prec": np.array(per_class_precisions),
                "per_class_rec": np.array(per_class_recalls),
                "per_class_f1": np.array(per_class_f1s),
            }
        )

    return results

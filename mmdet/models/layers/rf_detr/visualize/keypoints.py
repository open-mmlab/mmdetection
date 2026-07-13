# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Private keypoint visualization helpers for RF-DETR demos and diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from supervision import KeyPoints

from mmdet.models.layers.rf_detr.utilities.keypoints import precision_cholesky_to_pixel_covariance


def _copy_key_points(key_points: KeyPoints) -> KeyPoints:
    """Return a mutable copy of a Supervision ``KeyPoints`` object."""
    return KeyPoints(
        xy=key_points.xy.copy(),
        keypoint_confidence=(
            key_points.keypoint_confidence.copy() if key_points.keypoint_confidence is not None else None
        ),
        detection_confidence=(
            key_points.detection_confidence.copy() if key_points.detection_confidence is not None else None
        ),
        visible=key_points.visible.copy() if key_points.visible is not None else None,
        class_id=key_points.class_id.copy() if key_points.class_id is not None else None,
        data=dict(key_points.data),
    )


def _key_points_for_display(
    key_points: KeyPoints,
    *,
    keypoint_threshold: float = 0.0,
) -> KeyPoints:
    """Build ``KeyPoints`` for visualization from RF-DETR keypoint predictions.

    Args:
        key_points: RF-DETR keypoint prediction output.
        keypoint_threshold: Per-keypoint confidence threshold used to hide
            low-confidence points through ``key_points.visible``.

    Returns:
        A Supervision ``KeyPoints`` object ready for keypoint annotators.

    Raises:
        ValueError: If keypoints are present without per-point confidence.
    """
    key_points = _copy_key_points(key_points)

    if len(key_points) == 0:
        return key_points
    keypoint_confidence = key_points.keypoint_confidence
    if keypoint_confidence is None:
        raise ValueError("Expected RF-DETR keypoints to include per-keypoint confidence.")

    raw_precision = key_points.data.get("keypoint_precision_cholesky")
    raw_source_shape = key_points.data.get("source_shape")
    if "covariance" not in key_points.data and raw_precision is not None and raw_source_shape is not None:
        precision = np.asarray(raw_precision, dtype=np.float32)
        source_shape = np.asarray(raw_source_shape, dtype=np.float32)
        if precision.shape[:2] == key_points.xy.shape[:2] and source_shape.shape == (len(key_points), 2):
            key_points.data["covariance"] = precision_cholesky_to_pixel_covariance(
                precision_cholesky=precision,
                source_shape=source_shape,
            )

    visible = keypoint_confidence >= keypoint_threshold
    existing_visible = key_points.visible
    key_points.visible = visible if existing_visible is None else existing_visible & visible
    return key_points


def _keypoint_prediction_records(
    key_points: KeyPoints,
    *,
    image: str | Path | None = None,
    keypoint_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """Build flat keypoint prediction rows for notebook or terminal display.

    Args:
        key_points: RF-DETR keypoint prediction output.
        image: Optional image identifier added to each output row. Paths are
            represented by their file name.
        keypoint_threshold: Per-keypoint confidence threshold used when
            ``key_points.visible`` is not already populated.

    Returns:
        A list of row dictionaries, one per visible/non-zero keypoint.
    """
    keypoint_confidence = key_points.keypoint_confidence
    if keypoint_confidence is None:
        return []

    image_name = Path(image).name if image is not None else None
    existing_visible = key_points.visible
    visible = existing_visible if existing_visible is not None else keypoint_confidence >= keypoint_threshold
    detection_confidence = key_points.detection_confidence
    class_names = key_points.data.get("class_name")
    records: list[dict[str, Any]] = []
    for detection_index, xy in enumerate(key_points.xy):
        for keypoint_index, (point, confidence, is_visible) in enumerate(
            zip(xy, keypoint_confidence[detection_index], visible[detection_index], strict=True)
        ):
            if not is_visible or np.allclose(point, 0):
                continue
            class_name = None
            if isinstance(class_names, np.ndarray) and detection_index < len(class_names):
                class_name = str(class_names[detection_index])
            records.append(
                {
                    "image": image_name,
                    "detection_index": detection_index,
                    "class_id": int(key_points.class_id[detection_index]) if key_points.class_id is not None else None,
                    "class_name": class_name,
                    "detection_confidence": (
                        float(detection_confidence[detection_index]) if detection_confidence is not None else None
                    ),
                    "keypoint_index": keypoint_index,
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "keypoint_confidence": float(confidence),
                }
            )
    return records

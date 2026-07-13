# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from pathlib import Path

import numpy as np
from PIL import Image
from supervision import BoxAnnotator, Color, ColorLookup, ColorPalette, Detections, LabelAnnotator, Position

from mmdet.models.layers.rf_detr.utilities.logger import get_logger

logger = get_logger()


def save_gt_predictions_visualization(
    scenario_name: str,
    image_width: int,
    image_height: int,
    gt_boxes: list[list[float]],
    gt_class_ids: list[int],
    pred_boxes: list[list[float]],
    pred_class_ids: list[int],
    pred_confidences: list[float],
    pred_ious: list[float | None],
    save_dir: Path,
) -> None:
    """Save a visualization image showing both GT and prediction boxes.

    Boxes are labeled with class ID and confidence (for predictions). For predictions with known IoU, the IoU value is
    also shown.
    """
    from supervision import xywh_to_xyxy

    save_dir.mkdir(exist_ok=True)

    top_padding = 60
    image = np.zeros((image_height + top_padding, image_width, 3), dtype=np.uint8)
    scene: Image.Image = Image.fromarray(image)

    gt_boxes_offset = [[x, y + top_padding, w, h] for x, y, w, h in gt_boxes]
    pred_boxes_offset = [[x, y + top_padding, w, h] for x, y, w, h in pred_boxes]

    gt_xyxy = xywh_to_xyxy(np.array(gt_boxes_offset))
    pred_xyxy = xywh_to_xyxy(np.array(pred_boxes_offset))

    gt_detections = None
    pred_detections = None

    if len(gt_xyxy) > 0:
        gt_detections = Detections(
            xyxy=gt_xyxy,
            class_id=np.array(gt_class_ids),
        )

    if len(pred_xyxy) > 0:
        pred_detections = Detections(
            xyxy=pred_xyxy,
            class_id=np.array(pred_class_ids),
            confidence=np.array(pred_confidences),
        )

    # Index 0 is unused because class IDs start at 1
    gt_colors = ColorPalette(
        [
            Color(128, 128, 128),  # dummy color for index 0
            Color(0, 255, 100),
            Color(0, 200, 255),
        ]
    )
    pred_colors = ColorPalette(
        [
            Color(128, 128, 128),  # dummy color for index 0
            Color(255, 100, 50),
            Color(255, 50, 200),
        ]
    )

    gt_box_annotator = BoxAnnotator(color=gt_colors, thickness=3, color_lookup=ColorLookup.CLASS)
    pred_box_annotator = BoxAnnotator(color=pred_colors, thickness=3, color_lookup=ColorLookup.CLASS)

    gt_label_annotator = LabelAnnotator(
        color=gt_colors,
        text_color=Color.BLACK,
        text_scale=0.5,
        text_padding=3,
        text_position=Position.TOP_LEFT,
        color_lookup=ColorLookup.CLASS,
    )
    pred_label_annotator = LabelAnnotator(
        color=pred_colors,
        text_color=Color.BLACK,
        text_scale=0.5,
        text_padding=3,
        text_position=Position.TOP_RIGHT,
        color_lookup=ColorLookup.CLASS,
    )

    gt_labels = [f"c{class_id}" for class_id in gt_class_ids]

    pred_labels = []
    for class_id, conf, iou in zip(pred_class_ids, pred_confidences, pred_ious):
        if iou is not None:
            pred_labels.append(f"c{class_id}\nconf={conf:.3f}\niou={iou:.3f}")
        else:
            pred_labels.append(f"c{class_id}\nconf={conf:.3f}")

    if gt_detections is not None:
        scene = gt_box_annotator.annotate(scene=scene, detections=gt_detections)
        scene = gt_label_annotator.annotate(scene=scene, detections=gt_detections, labels=gt_labels)
    if pred_detections is not None:
        scene = pred_box_annotator.annotate(scene=scene, detections=pred_detections)
        scene = pred_label_annotator.annotate(scene=scene, detections=pred_detections, labels=pred_labels)

    scene.save(save_dir / f"{scenario_name}.png")
    logger.info(f"Saved visualization to {save_dir}/{scenario_name}.png")

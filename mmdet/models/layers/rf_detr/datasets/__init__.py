# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any

import torchvision
from torch.utils.data import Dataset, Subset

from mmdet.models.layers.rf_detr.datasets._keypoint_schema import infer_coco_keypoint_schema as infer_coco_keypoint_schema
from mmdet.models.layers.rf_detr.datasets._keypoint_schema import infer_yolo_keypoint_schema as infer_yolo_keypoint_schema
from mmdet.models.layers.rf_detr.datasets.coco import build_coco, build_roboflow_from_coco
from mmdet.models.layers.rf_detr.datasets.o365 import build_o365
from mmdet.models.layers.rf_detr.datasets.yolo import YoloDetection, build_roboflow_from_yolo


def get_coco_api_from_dataset(dataset: Dataset[Any]) -> Any | None:
    for _ in range(10):
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    if isinstance(dataset, YoloDetection):
        return dataset.coco
    return None


def detect_roboflow_format(dataset_dir: Path) -> str:
    """Detect if a Roboflow dataset is in COCO or YOLO format.

    Args:
        dataset_dir: Path to the Roboflow dataset root directory

    Returns:
        'coco' if COCO format detected, 'yolo' if YOLO format detected

    Raises:
        ValueError: If neither format is detected
    """
    # Check for COCO format: look for _annotations.coco.json in train folder
    coco_annotation = dataset_dir / "train" / "_annotations.coco.json"
    if coco_annotation.exists():
        return "coco"

    # Check for YOLO format: look for data.yaml or data.yml and train/images folder
    yolo_data_file_yaml = dataset_dir / "data.yaml"
    yolo_data_file_yml = dataset_dir / "data.yml"
    yolo_images_dir = dataset_dir / "train" / "images"
    if (yolo_data_file_yaml.exists() or yolo_data_file_yml.exists()) and yolo_images_dir.exists():
        return "yolo"

    raise ValueError(
        f"Could not detect dataset format in {dataset_dir}. "
        f"Expected either COCO format (train/_annotations.coco.json) "
        f"or YOLO format (data.yaml or data.yml + train/images/)"
    )


def build_roboflow(image_set: str, args: Any, resolution: int) -> Dataset[Any]:
    """Build a Roboflow dataset, auto-detecting COCO or YOLO format.

    This function detects the dataset format and delegates to the appropriate builder function.
    """
    root = Path(args.dataset_dir)
    assert root.exists(), f"provided Roboflow path {root} does not exist"

    dataset_format = detect_roboflow_format(root)

    if dataset_format == "coco":
        return build_roboflow_from_coco(image_set, args, resolution)
    return build_roboflow_from_yolo(image_set, args, resolution)


def build_dataset(image_set: str, args: Any, resolution: int) -> Dataset[Any]:
    if args.dataset_file == "coco":
        return build_coco(image_set, args, resolution)
    if args.dataset_file == "o365":
        return build_o365(image_set, args, resolution)
    if args.dataset_file == "roboflow":
        return build_roboflow(image_set, args, resolution)
    if args.dataset_file == "yolo":
        return build_roboflow_from_yolo(image_set, args, resolution)
    raise ValueError(f"dataset {args.dataset_file} not supported")

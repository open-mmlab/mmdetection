# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
"""Dataset file for Object365."""

from pathlib import Path
from typing import Any

from PIL import Image

from mmdet.models.layers.rf_detr.datasets.coco import CocoDetection, make_coco_transforms, make_coco_transforms_square_div_64
from mmdet.models.layers.rf_detr.utilities.logger import get_logger

# O365 contains images larger than PIL's default 178M-pixel limit.
# Set a generous but finite cap (2 billion pixels ~ a 45k x 45k image) instead of
# disabling the guard entirely, so PIL still protects against decompression bombs.
Image.MAX_IMAGE_PIXELS = 2_000_000_000

logger = get_logger()


def build_o365_raw(image_set: str, args: Any, resolution: int) -> CocoDetection:
    root = Path(getattr(args, "dataset_dir", None) or args.coco_path)
    PATHS = {  # noqa: N806
        "train": (root, root / "zhiyuan_objv2_train_val_wo_5k.json"),
        "val": (root, root / "zhiyuan_objv2_minival5k.json"),
    }
    img_folder, ann_file = PATHS[image_set]

    from mmdet.models.layers.rf_detr.datasets.kornia_transforms import resolve_augmentation_backend

    square_resize_div_64 = getattr(args, "square_resize_div_64", False)
    augmentation_backend = getattr(args, "augmentation_backend", "cpu")
    resolved_backend = resolve_augmentation_backend(augmentation_backend)

    if resolved_backend != "cpu":
        logger.warning(
            "O365 dataset does not support custom aug_config in Phase 1 GPU augmentation; "
            "Albumentations augmentation is skipped and normalization runs on GPU. "
            "Pass augmentation_backend='cpu' for full CPU augmentation pipeline with O365."
        )
    gpu_postprocess = resolved_backend != "cpu"

    if square_resize_div_64:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms_square_div_64(
                image_set,
                resolution,
                multi_scale=args.multi_scale,
                expanded_scales=args.expanded_scales,
                gpu_postprocess=gpu_postprocess,
            ),
        )
    else:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms(
                image_set,
                resolution,
                multi_scale=args.multi_scale,
                expanded_scales=args.expanded_scales,
                gpu_postprocess=gpu_postprocess,
            ),
        )
    return dataset


def build_o365(image_set: str, args: Any, resolution: int) -> CocoDetection:
    if image_set == "train":
        train_ds = build_o365_raw("train", args, resolution=resolution)
        return train_ds
    if image_set == "val":
        val_ds = build_o365_raw("val", args, resolution=resolution)
        return val_ds
    raise ValueError(f"Unknown image_set: {image_set}")

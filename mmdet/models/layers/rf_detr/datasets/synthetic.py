# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
#
#
"""Synthetic dataset generation with COCO formatting."""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from supervision import Color, Detections, box_iou_batch, draw_filled_polygon
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSplitRatios:
    """Dataclass for dataset split ratios.

    Attributes:
        train: Ratio for training set (default: 0.7)
        val: Ratio for validation set (default: 0.2)
        test: Ratio for test set (default: 0.1)

    Raises:
        ValueError: If ratios are negative or sum is not approximately 1.0.
    """

    train: float = 0.7
    val: float = 0.2
    test: float = 0.1

    def __post_init__(self):
        """Validate that ratios sum to approximately 1.0 and are non-negative."""
        total = self.train + self.val + self.test
        if any(r < 0 for r in [self.train, self.val, self.test]):
            raise ValueError(
                f"Split ratios must be non-negative, got train={self.train}, val={self.val}, test={self.test}"
            )
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary, filtering out zero ratios."""
        return {k: v for k, v in {"train": self.train, "val": self.val, "test": self.test}.items() if v > 0}


# Default split ratios instance
DEFAULT_SPLIT_RATIOS = DatasetSplitRatios()  # 70/20/10 split


# Type alias for split ratios parameter
SplitRatiosType = DatasetSplitRatios | tuple[float, ...] | dict[str, float]


def _normalize_split_ratios(split_ratios: SplitRatiosType) -> dict[str, float]:
    """Normalize split ratios parameter to a dictionary.

    Args:
        split_ratios: Can be:
            - DatasetSplitRatios dataclass instance
            - Tuple of floats (e.g., (0.7, 0.2, 0.1) for train/val/test)
            - Dictionary (legacy support)

    Returns:
        Dictionary mapping split names to ratios.

    Raises:
        ValueError: If split ratios are invalid.
    """
    if isinstance(split_ratios, DatasetSplitRatios):
        return split_ratios.to_dict()

    if isinstance(split_ratios, tuple):
        if len(split_ratios) == 2:
            result = {"train": split_ratios[0], "val": split_ratios[1]}
        elif len(split_ratios) == 3:
            result = {"train": split_ratios[0], "val": split_ratios[1], "test": split_ratios[2]}
        else:
            raise ValueError(f"Split ratios tuple must have 2 or 3 elements, got {len(split_ratios)}")

        # Validate tuple ratios are non-negative and sum to approximately 1.0
        if any(ratio < 0 for ratio in split_ratios):
            raise ValueError(f"Split ratios must be non-negative, got {split_ratios}")
        total = sum(split_ratios)
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        return result

    if isinstance(split_ratios, dict):
        # Validate that ratios are non-negative and sum to approximately 1.0
        if any(value < 0 for value in split_ratios.values()):
            raise ValueError(f"Split ratios must be non-negative, got {split_ratios}")
        total = sum(split_ratios.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        return split_ratios

    raise TypeError(f"split_ratios must be DatasetSplitRatios, tuple, or dict, got {type(split_ratios)}")


# Available shapes for synthetic dataset generation
SYNTHETIC_SHAPES = ["square", "triangle", "circle"]
# Available colors for synthetic dataset generation (RGB format)
SYNTHETIC_COLORS = {"red": Color.RED, "green": Color.GREEN, "blue": Color.BLUE}


def draw_synthetic_shape(
    img: np.ndarray, shape: str, color: Color, center: tuple[int, int], size: int
) -> tuple[np.ndarray, list[float]]:
    """Draw a geometric shape on an image and return its COCO polygon.

    The polygon is computed first, then used for both rendering and annotation, so the two are always identical.

    Args:
        img: Input image array to draw on.
        shape: Shape to draw (``"square"``, ``"triangle"``, or ``"circle"``).
        color: supervision Color object.
        center: Center position ``(cx, cy)``.
        size: Size of the shape.

    Returns:
        Tuple of ``(image_with_shape, polygon)`` where ``polygon`` is a flat list ``[x1, y1, x2, y2, …]`` suitable for
        the COCO ``segmentation`` field.  Returns an empty polygon list for unknown shape names.
    """
    cx, cy = center
    half_size = size // 2

    if shape == "square":
        x1, y1 = cx - half_size, cy - half_size
        x2, y2 = cx + half_size, cy + half_size
        pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    elif shape == "triangle":
        # Apex at cy - size//2, base at cy + size//4. Total height = 0.75 * size.
        height = int(size * 0.75)
        pts = [
            [cx, cy - 2 * height // 3],
            [cx - half_size, cy + height // 3],
            [cx + half_size, cy + height // 3],
        ]
    elif shape == "circle":
        r = half_size
        n_pts = 32
        pts = [
            [int(cx + r * math.cos(2 * math.pi * i / n_pts)), int(cy + r * math.sin(2 * math.pi * i / n_pts))]
            for i in range(n_pts)
        ]
    else:
        return img, []

    img = draw_filled_polygon(scene=img, polygon=np.array(pts, dtype=np.int32), color=color)
    polygon = [float(v) for pt in pts for v in pt]
    return img, polygon


def calculate_boundary_overlap(bbox: np.ndarray, img_size: int) -> float:
    """Calculate how much of a bounding box is outside the image boundaries.

    Args:
        bbox: Bounding box in [x_min, y_min, x_max, y_max] format.
        img_size: Size of the image.

    Returns:
        Overlap fraction in ``[0, 1]``: ``0.0`` means the box is fully inside the image; ``1.0`` means it is fully
        outside.
    """
    x_min, y_min, x_max, y_max = bbox

    inside_x_min = max(x_min, 0)
    inside_y_min = max(y_min, 0)
    inside_x_max = min(x_max, img_size)
    inside_y_max = min(y_max, img_size)

    if inside_x_max > inside_x_min and inside_y_max > inside_y_min:
        inside_area = (inside_x_max - inside_x_min) * (inside_y_max - inside_y_min)
    else:
        inside_area = 0.0

    total_area = (x_max - x_min) * (y_max - y_min)
    return 1.0 - (inside_area / total_area) if total_area > 0 else 0.0


def generate_synthetic_sample(
    img_size: int,
    min_objects: int,
    max_objects: int,
    class_mode: Literal["shape", "color"],
    min_size_ratio: float = 0.1,
    max_size_ratio: float = 0.3,
    overlap_threshold: float = 0.1,
) -> tuple[np.ndarray, Detections]:
    """Generate a single synthetic image and its detections.

    Args:
        img_size: Side length of the square output image.
        min_objects: Minimum number of objects to attempt placing.
        max_objects: Maximum number of objects to attempt placing.
        class_mode: ``"shape"`` assigns class IDs by shape type;
            ``"color"`` assigns class IDs by colour.
        min_size_ratio: Minimum object size as a fraction of ``img_size``.
        max_size_ratio: Maximum object size as a fraction of ``img_size``.
        overlap_threshold: Maximum allowed IoU between any two objects before
            a placement attempt is rejected.

    Returns:
        Tuple of ``(image, detections)`` where ``image`` is an ``(img_size, img_size, 3)`` uint8 array and
        ``detections`` is an :class:`Detections` instance whose ``data["polygons"]`` field contains one flat ``[x1,
        y1, x2, y2, …]`` polygon list per detection, matching the geometry returned by :func:`draw_synthetic_shape`.
    """
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 128
    color_names = list(SYNTHETIC_COLORS.keys())
    num_objects = random.randint(min_objects, max_objects)

    xyxys = []
    class_ids = []
    polygons: list[list[float]] = []
    failed_attempts = 0
    max_failed_attempts = 3  # Allow some failures before reducing target count

    for _ in range(num_objects):
        shape = random.choice(SYNTHETIC_SHAPES)
        color_name = random.choice(color_names)
        color = SYNTHETIC_COLORS[color_name]

        if class_mode == "shape":
            category_id = SYNTHETIC_SHAPES.index(shape)
        else:
            category_id = color_names.index(color_name)

        min_size = max(10, int(img_size * min_size_ratio))
        max_size = max(min_size + 1, int(img_size * max_size_ratio))

        placed = False
        for _ in range(100):  # max attempts per object
            obj_size = random.randint(min_size, max_size)
            cx = random.randint(obj_size // 2, img_size - obj_size // 2)
            cy = random.randint(obj_size // 2, img_size - obj_size // 2)

            # [x_min, y_min, x_max, y_max]
            bbox = np.array(
                [float(cx - obj_size / 2), float(cy - obj_size / 2), float(cx + obj_size / 2), float(cy + obj_size / 2)]
            )

            if calculate_boundary_overlap(bbox, img_size) > 0.05:
                continue

            if len(xyxys) > 0:
                ious = box_iou_batch(np.array([bbox]), np.array(xyxys))[0]
                if np.any(ious > overlap_threshold):
                    continue

            img, polygon = draw_synthetic_shape(img, shape, color, (cx, cy), obj_size)

            # Derive bbox directly from the rendered polygon to ensure consistency
            polygon_array = np.asarray(polygon, dtype=float).reshape(-1, 2)
            poly_x_min = float(np.min(polygon_array[:, 0]))
            poly_y_min = float(np.min(polygon_array[:, 1]))
            poly_x_max = float(np.max(polygon_array[:, 0]))
            poly_y_max = float(np.max(polygon_array[:, 1]))
            bbox_from_polygon = np.array([poly_x_min, poly_y_min, poly_x_max, poly_y_max], dtype=float)

            xyxys.append(bbox_from_polygon)
            class_ids.append(category_id)
            polygons.append(polygon)
            placed = True
            break

        # Track failed placements; stop early if too crowded
        if not placed:
            failed_attempts += 1
            if failed_attempts >= max_failed_attempts:
                break

    polygon_data = np.empty(len(class_ids), dtype=object)
    for i, poly in enumerate(polygons):
        polygon_data[i] = poly

    detections = Detections(
        xyxy=np.array(xyxys) if xyxys else np.empty((0, 4)),
        class_id=np.array(class_ids) if class_ids else np.empty((0,), dtype=int),
        data={"polygons": polygon_data},
    )
    return img, detections


def _calculate_polygon_area(polygon: list[float]) -> float:
    """Calculate polygon area from COCO-style flat coordinates."""
    if len(polygon) < 6 or len(polygon) % 2 != 0:
        return 0.0

    points = np.asarray(polygon, dtype=float).reshape(-1, 2)
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    return float(0.5 * abs(np.dot(x_coords, np.roll(y_coords, -1)) - np.dot(y_coords, np.roll(x_coords, -1))))


def _write_coco_json(
    annotations_path: Path,
    classes: list[str],
    file_paths: list[str],
    detections_list: list[Detections],
    img_size: int,
    with_segmentation: bool = False,
) -> None:
    """Write a synthetic COCO JSON file.

    Category IDs use sparse 1-based encoding (index * 2 + 1 → 1, 3, 5, …) so synthetic data exercises the same
    ``cat2label`` remapping path that real COCO datasets use.

    Args:
        annotations_path: Destination path for the JSON file.
        classes: Ordered list of class names.
        file_paths: Ordered list of absolute image file paths (one per image).
        detections_list: Detections for each image in the same order.
        img_size: Side length of the square images (width = height = img_size).
        with_segmentation: When ``True`` each annotation includes a
            ``segmentation`` polygon taken from ``detections.data["polygons"]`` (populated by
            :func:`generate_synthetic_sample`).  When ``False`` the field is an empty list.

    Raises:
        ValueError: If ``file_paths`` and ``detections_list`` have different
            lengths.
        ValueError: If ``with_segmentation=True`` and a detections entry has
            no ``"polygons"`` key in its ``data`` dict.
        ValueError: If ``with_segmentation=True`` and the ``"polygons"`` array
            has fewer entries than there are detections for that image.
        ValueError: If any detection has a ``class_id`` outside the range
            ``[0, len(classes))``.
    """
    if len(file_paths) != len(detections_list):
        raise ValueError(
            "file_paths and detections_list must have the same length, "
            f"but got {len(file_paths)} and {len(detections_list)}"
        )

    categories = [{"id": idx * 2 + 1, "name": name, "supercategory": "synthetic"} for idx, name in enumerate(classes)]
    images_list = []
    annotations_list = []
    ann_id = 1

    for img_id, (file_path, detections) in enumerate(zip(file_paths, detections_list), start=1):
        images_list.append(
            {
                "id": img_id,
                "file_name": Path(file_path).name,
                "width": img_size,
                "height": img_size,
            }
        )
        if with_segmentation:
            polygon_data = detections.data.get("polygons")
            if polygon_data is None:
                raise ValueError(
                    f"with_segmentation=True but no 'polygons' found in detections.data "
                    f"for image index {img_id} (file: {file_path})"
                )
            if len(polygon_data) < len(detections):
                raise ValueError(
                    "with_segmentation=True requires a polygon entry for every detection (one per detection index), "
                    f"but got only {len(polygon_data)} polygon entries for {len(detections)} detections "
                    f"in image index {img_id} (file: {file_path})"
                )
        else:
            polygon_data = np.empty(0, dtype=object)
        for det_idx in range(len(detections)):
            x1, y1, x2, y2 = (float(v) for v in detections.xyxy[det_idx])
            w, h_box = x2 - x1, y2 - y1
            class_id = int(detections.class_id[det_idx])
            if class_id < 0 or class_id >= len(classes):
                raise ValueError(
                    f"Invalid class_id {class_id} for detection index {det_idx} "
                    f"in image index {img_id} (file: {file_path}); "
                    f"expected 0 <= class_id < {len(classes)}"
                )
            category_id = class_id * 2 + 1
            annotation_area = w * h_box
            if with_segmentation:
                poly = polygon_data[det_idx] if det_idx < len(polygon_data) else None
                if poly is not None and hasattr(poly, "__len__") and len(poly) > 0:
                    poly_list = [float(value) for value in poly]
                    segmentation = [poly_list]
                    polygon_area = _calculate_polygon_area(poly_list)
                    if polygon_area > 0.0:
                        annotation_area = polygon_area
                else:
                    segmentation = []
            else:
                segmentation = []
            annotations_list.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, w, h_box],
                    "area": annotation_area,
                    "iscrowd": 0,
                    "segmentation": segmentation,
                }
            )
            ann_id += 1

    with open(annotations_path, "w") as fh:
        json.dump({"images": images_list, "annotations": annotations_list, "categories": categories}, fh)


def generate_coco_dataset(
    output_dir: str,
    num_images: int,
    img_size: int = 640,
    class_mode: Literal["shape", "color"] = "shape",
    min_objects: int = 1,
    max_objects: int = 10,
    split_ratios: SplitRatiosType = DEFAULT_SPLIT_RATIOS,
    with_segmentation: bool = False,
) -> None:
    """Generate a full synthetic dataset in COCO format.

    Args:
        output_dir: Directory where the dataset will be saved.
        num_images: Total number of images to generate.
        img_size: Size of the square images.
        class_mode: Classification mode - "shape" or "color" (default: "shape").
        min_objects: Minimum objects per image.
        max_objects: Maximum objects per image.
        split_ratios: Dataset split ratios. Can be:
            - SplitRatios dataclass instance (default: 70/20/10 split)
            - Tuple of 2 floats for train/val (e.g., (0.8, 0.2))
            - Tuple of 3 floats for train/val/test (e.g., (0.7, 0.2, 0.1))
            - Dictionary (legacy support, e.g., {"train": 0.7, "val": 0.2, "test": 0.1})
        with_segmentation: If ``True``, include COCO polygon ``segmentation``
            fields derived from the exact geometry of each drawn shape. Requires the COCO dataset reader to be loaded
            with ``include_masks=True`` (i.e. ``args.segmentation_head=True``).
    """
    # Normalize split_ratios to dictionary
    split_ratios_dict = _normalize_split_ratios(split_ratios)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if class_mode == "shape":
        classes = SYNTHETIC_SHAPES
    else:
        classes = list(SYNTHETIC_COLORS.keys())

    # Shuffle indices for splits
    all_indices = list(range(num_images))
    random.shuffle(all_indices)

    start_idx = 0
    split_items = list(split_ratios_dict.items())
    for split_idx, (split, ratio) in enumerate(split_items):
        if split_idx == len(split_items) - 1:
            # Last split absorbs any remainder to avoid silently losing images
            num_split = len(all_indices) - start_idx
        else:
            num_split = int(num_images * ratio)
            if num_split == 0 and ratio > 0:
                num_split = 1
        split_indices = all_indices[start_idx : start_idx + num_split]
        start_idx += num_split

        if not split_indices:
            continue

        # Images and annotations should be in the same directory for COCO format
        split_dir = output_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        annotations_path = split_dir / "_annotations.coco.json"

        file_paths_ordered: list[str] = []
        detections_ordered: list[Detections] = []

        logger.info(f"Generating {split} split with {len(split_indices)} images...")
        for i in tqdm(split_indices, desc=f"Generating {split} split"):
            img, detections = generate_synthetic_sample(
                img_size,
                min_objects,
                max_objects,
                class_mode,
            )

            file_name = f"{i:06d}.jpg"
            file_path = str(split_dir / file_name)
            # ``supervision.draw_filled_polygon`` paints colors in BGR order (via ``cv2.fillPoly``);
            # flip the last axis so PIL writes a true RGB JPEG that downstream loaders read correctly.
            Image.fromarray(img[..., ::-1]).save(file_path)

            file_paths_ordered.append(file_path)
            detections_ordered.append(detections)

        _write_coco_json(annotations_path, classes, file_paths_ordered, detections_ordered, img_size, with_segmentation)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic COCO dataset")
    parser.add_argument("--output", type=str, default="synthetic_dataset", help="Output directory")
    parser.add_argument("--num_images", type=int, default=100, help="Total number of images")
    parser.add_argument("--img_size", type=int, default=640, help="Image size (square)")
    parser.add_argument("--mode", type=str, choices=["shape", "color"], default="shape", help="Classification mode")

    args = parser.parse_args()
    generate_coco_dataset(args.output, args.num_images, args.img_size, args.mode)

# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import yaml

if TYPE_CHECKING:
    from supervision import Detections
from PIL import Image, ImageDraw
from torchvision.datasets import VisionDataset

from mmdet.models.layers.rf_detr.datasets._keypoint_schema import (
    YoloKeypointSchema,
    _extract_yolo_class_names_from_data,
    _load_yaml_mapping,
    infer_yolo_keypoint_schema,
)
from mmdet.models.layers.rf_detr.datasets.coco import (
    _resolve_runtime_augmentation_backend,
    make_coco_transforms,
    make_coco_transforms_square_div_64,
)
from mmdet.models.layers.rf_detr.utilities.logger import get_logger

logger = get_logger()

REQUIRED_YOLO_YAML_FILES = ["data.yaml", "data.yml"]
_VALID_VAL_DIR_NAMES = ("valid", "val")
REQUIRED_DATA_SUBDIRS = ["images", "labels"]
YOLO_IMAGE_EXTENSIONS = {".bmp", ".dng", ".jpg", ".jpeg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def _parse_yolo_box(values: list[str]) -> np.ndarray:
    """Parse a YOLO center-width-height box into relative XYXY coordinates."""
    x_center, y_center, width, height = values
    return np.array(
        [
            float(x_center) - float(width) / 2,
            float(y_center) - float(height) / 2,
            float(x_center) + float(width) / 2,
            float(y_center) + float(height) / 2,
        ],
        dtype=np.float32,
    )


def _box_to_polygon(box: np.ndarray) -> np.ndarray:
    """Convert a relative XYXY box into a 4-corner polygon."""
    return np.array(
        [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]],
        dtype=np.float32,
    )


def _parse_yolo_polygon(values: list[str]) -> np.ndarray:
    """Parse a flattened YOLO polygon into relative XY points."""
    return np.array(values, dtype=np.float32).reshape(-1, 2)


def _polygon_to_mask(polygon: np.ndarray, resolution_wh: tuple[int, int]) -> np.ndarray:
    """Rasterize a polygon into a dense boolean mask.

    TODO: remove once supervision ships a direct CompactMask.from_polygon factory;
    at that point the dense intermediate array is no longer needed.
    """
    width, height = resolution_wh
    mask = Image.new("L", (width, height), 0)
    if polygon.size > 0:
        ImageDraw.Draw(mask).polygon([tuple(point) for point in polygon.tolist()], fill=1)
    return np.array(mask, dtype=bool)


def _polygons_to_masks(polygons: tuple[np.ndarray, ...], resolution_wh: tuple[int, int]) -> np.ndarray:
    """Rasterize per-instance polygons into an ``(N, H, W)`` boolean array.

    TODO: remove once supervision ships a direct CompactMask.from_polygon factory;
    at that point the dense intermediate array is no longer needed.
    """
    if len(polygons) == 0:
        width, height = resolution_wh
        return np.zeros((0, height, width), dtype=bool)
    return np.stack([_polygon_to_mask(polygon, resolution_wh) for polygon in polygons])


def _list_yolo_image_paths(images_directory_path: str) -> list[str]:
    """List YOLO image files in a stable order."""
    return sorted(
        str(path)
        for path in Path(images_directory_path).iterdir()
        if path.is_file() and path.suffix.lower() in YOLO_IMAGE_EXTENSIONS
    )


def _extract_yolo_class_names(data_file: str) -> list[str]:
    """Read class names from a YOLO ``data.yaml`` file."""
    path = Path(data_file)
    data = _load_yaml_mapping(path)
    return _extract_yolo_class_names_from_data(data, path)


@dataclass(frozen=True)
class _LazyYoloSample:
    """Lightweight per-image YOLO metadata with polygons kept lazy until fetch time.

    Note: ``frozen=True`` prevents field *reassignment* but does NOT prevent
    in-place mutation of ``np.ndarray`` fields (e.g. ``sample.xyxy[0] = 999.0`` would silently succeed).  This is safe
    across DataLoader workers because each worker receives a pickled copy of the dataset.
    """

    image_path: str
    width: int
    height: int
    xyxy: np.ndarray
    class_id: np.ndarray
    polygons: tuple[np.ndarray, ...]
    keypoints: np.ndarray

    def to_detections(self) -> Detections:
        """Materialize the current sample as a supervision ``Detections`` object."""
        from supervision import Detections

        if len(self.class_id) == 0:
            return Detections.empty()
        if len(self.polygons) == 0:
            # Detection-only path: no masks were computed, return bare boxes.
            return Detections(class_id=self.class_id, xyxy=self.xyxy)
        # TODO: once supervision v0.28 ships CompactMask, wrap the dense result:
        #   compact = sv.CompactMask.from_dense(mask, self.xyxy, (self.height, self.width))
        #   return Detections(..., mask=compact)
        # CompactMask stores crop-RLE instead of a full H×W bool array, reducing memory
        # at the detections level for large images with sparse objects.
        # Note: _polygon_to_mask / _polygons_to_masks remain required as the intermediate
        # rasterization step until supervision provides a direct from_polygon factory.
        mask = _polygons_to_masks(self.polygons, (self.width, self.height))
        return Detections(class_id=self.class_id, xyxy=self.xyxy, mask=mask)


class _LazyYoloDetectionDataset:
    """Lazy YOLO dataset that defers dense mask rasterization until ``__getitem__``."""

    def __init__(self, classes: list[str], samples: list[_LazyYoloSample]) -> None:
        self.classes = classes
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[str, np.ndarray, Detections]:
        sample = self._samples[idx]
        try:
            with Image.open(sample.image_path) as image:
                rgb_image = np.array(image.convert("RGB"))
        except (FileNotFoundError, OSError, Image.UnidentifiedImageError) as exc:
            raise ValueError(f"Could not read image from path: {sample.image_path}") from exc
        return sample.image_path, rgb_image, sample.to_detections()

    def get_image_info(self, idx: int) -> _LazyYoloSample:
        """Return lightweight metadata without loading pixels or dense masks."""
        return self._samples[idx]


def _parse_yolo_label_line(
    values: list[str],
    line_num: int,
    label_path: Path,
    num_classes: int,
    width: int,
    height: int,
    *,
    parse_polygons: bool = True,
) -> tuple[int, np.ndarray, np.ndarray | None]:
    """Parse one YOLO label line and return ``(class_id, xyxy_px, polygon_px)``.

    Args:
        values: Whitespace-split fields from the label line.
        line_num: 1-based line number (for error messages).
        label_path: Path to the label file (for error messages).
        num_classes: Total number of classes in the dataset (used for range check).
        width: Image width in pixels.
        height: Image height in pixels.
        parse_polygons: When ``False`` the pixel-space polygon array is not
            computed or returned (``polygon_px`` will be ``None``).  Set to ``False`` on the detection-only path to
            avoid allocating polygon arrays that would immediately be discarded.

    Returns:
        Tuple of ``(class_id, xyxy_px, polygon_px)`` where coordinates are in pixel space.  ``polygon_px`` is ``None``
        when ``parse_polygons=False``.

    Raises:
        ValueError: If the line is malformed or the class ID is out of range.
    """
    if len(values) < 5:
        raise ValueError(
            f"Malformed label in {str(label_path)!r} at line {line_num}: "
            f"expected 5 (bbox) fields or ≥ 7 fields for polygons "
            f"(class_id + at least 3 (x, y) points), got {len(values)}."
        )
    if len(values) > 5 and len(values[1:]) % 2 != 0:
        raise ValueError(
            f"Malformed polygon in {str(label_path)!r} at line {line_num}: "
            f"polygon coordinates must be paired (x, y) values, "
            f"but got {len(values[1:])} coordinate values (odd count)."
        )
    try:
        cid = int(values[0])
    except ValueError as exc:
        raise ValueError(
            f"Label {str(label_path)!r} line {line_num}: invalid class ID {values[0]!r} (must be an integer)."
        ) from exc
    # num_classes equals len(class_names) which _extract_yolo_class_names guarantees
    # is a contiguous 0..N-1 range.  This assumption must remain consistent with the
    # class-name parser: accepting sparse keys there (e.g. {0: "cat", 2: "dog"} → 2
    # classes) would cause valid label files using the original IDs to be rejected here.
    if cid < 0 or cid >= num_classes:
        raise ValueError(
            f"Label {str(label_path)!r} line {line_num}: "
            f"class ID {cid} is out of range for dataset with {num_classes} classes "
            f"(valid range 0\u2013{num_classes - 1})."
        )
    if len(values) == 5:
        box = _parse_yolo_box(values[1:])
        # Skip polygon creation on the detection path — only the bbox is needed.
        polygon: np.ndarray | None = _box_to_polygon(box) if parse_polygons else None
    else:
        try:
            _raw_polygon = _parse_yolo_polygon(values[1:])
        except ValueError as exc:
            raise ValueError(
                f"Malformed polygon in {str(label_path)!r} at line {line_num}: "
                f"could not parse coordinate values as floats."
            ) from exc
        box = np.array(
            [
                np.min(_raw_polygon[:, 0]),
                np.min(_raw_polygon[:, 1]),
                np.max(_raw_polygon[:, 0]),
                np.max(_raw_polygon[:, 1]),
            ],
            dtype=np.float32,
        )
        # On the detection path, _raw_polygon was only needed for bbox extraction;
        # skip the pixel-space conversion to avoid a redundant allocation.
        polygon = _raw_polygon if parse_polygons else None
    xyxy_px = box * np.array([width, height, width, height], dtype=np.float32)
    if polygon is None:
        return cid, xyxy_px, None
    polygon_px = polygon * np.array([width, height], dtype=np.float32)
    polygon_px[:, 0] = np.clip(polygon_px[:, 0], 0.0, float(width - 1))
    polygon_px[:, 1] = np.clip(polygon_px[:, 1], 0.0, float(height - 1))
    return cid, xyxy_px, polygon_px.astype(np.float32)


def _parse_yolo_pose_label_line(
    values: list[str],
    line_num: int,
    label_path: Path,
    num_classes: int,
    width: int,
    height: int,
    *,
    num_keypoints: int,
    keypoint_dim: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Parse one Ultralytics YOLO pose row into pixel boxes and COCO-style keypoints."""
    expected_fields = 5 + num_keypoints * keypoint_dim
    if len(values) != expected_fields:
        hint = (
            " This looks like a detection-only label row (5 fields). "
            "Check whether the dataset mixes detection and pose annotations "
            "or whether the kpt_shape in data.yaml is correct."
            if len(values) == 5 and num_keypoints > 0
            else ""
        )
        raise ValueError(
            f"Malformed YOLO pose label in {str(label_path)!r} at line {line_num}: "
            f"expected {expected_fields} fields from kpt_shape=[{num_keypoints}, {keypoint_dim}], "
            f"got {len(values)}.{hint}"
        )

    cid, xyxy_px, _ = _parse_yolo_label_line(
        values[:5],
        line_num,
        label_path,
        num_classes,
        width,
        height,
        parse_polygons=False,
    )
    try:
        raw_keypoints = np.asarray(values[5:], dtype=np.float32).reshape(num_keypoints, keypoint_dim)
    except ValueError as exc:
        raise ValueError(
            f"Malformed YOLO pose label in {str(label_path)!r} at line {line_num}: "
            "could not parse keypoint values as floats."
        ) from exc

    if not np.isfinite(raw_keypoints).all():
        raise ValueError(f"Malformed YOLO pose label in {str(label_path)!r} at line {line_num}: non-finite keypoint.")
    xy = raw_keypoints[:, :2]

    keypoints = np.zeros((num_keypoints, 3), dtype=np.float32)
    if keypoint_dim == 3:
        # v is authoritative for absent/present; clamp OOB coords to image edge.
        visibility = raw_keypoints[:, 2]
        if np.any((visibility < 0.0) | (visibility > 2.0)):
            raise ValueError(
                f"Malformed YOLO pose label in {str(label_path)!r} at line {line_num}: "
                "keypoint visibility values must be in [0, 2]."
            )
        np.clip(xy, 0.0, 1.0, out=xy)
        keypoints[:, 2] = visibility
    else:
        # Ultralytics dim-2 format: absent keypoints are marked with negative
        # coordinates (any coord < 0 → absent).  Detect BEFORE clamping so that
        # a keypoint like (-0.1, 0.5) is not clamped to (0.0, 0.5) and
        # mistakenly treated as a present keypoint at the left image edge.
        absent_2d = (xy[:, 0] < 0.0) | (xy[:, 1] < 0.0)
        np.clip(xy, 0.0, 1.0, out=xy)
        # Zero coords for absent keypoints so the (0, 0) absent sentinel is set.
        xy[absent_2d, :] = 0.0
        present = ~((xy[:, 0] == 0.0) & (xy[:, 1] == 0.0))
        keypoints[present, 2] = 2.0

    keypoints[:, 0] = xy[:, 0] * float(width)
    keypoints[:, 1] = xy[:, 1] * float(height)

    absent = keypoints[:, 2] <= 0.0
    keypoints[absent, :2] = 0.0
    return cid, xyxy_px, keypoints


def _build_yolo_samples(
    img_folder: str,
    lb_folder: str,
    data_file: str,
    *,
    include_polygons: bool,
    include_keypoints: bool = False,
    keypoint_schema: YoloKeypointSchema | None = None,
) -> tuple[list[str], list[_LazyYoloSample]]:
    """Build the class list and sample list shared by both YOLO builder functions.

    Iterates over every image in ``img_folder``, reads image dimensions via PIL (header-only, no full decode), and
    parses the matching ``.txt`` label file when present.  Images without a label file are included as *background*
    samples with empty detections.

    Args:
        img_folder: Path to the directory containing images.
        lb_folder: Path to the directory containing YOLO ``.txt`` label files.
        data_file: Path to the ``data.yaml`` / ``data.yml`` file with class names.
        include_polygons: When ``True`` polygon coordinates are stored in each
            :class:`_LazyYoloSample` (segmentation path).  When ``False`` polygon coordinates returned by
            :func:`_parse_yolo_label_line` are discarded and ``polygons=()`` is stored instead (detection-only path).
            Mutually exclusive with ``include_keypoints``.
        include_keypoints: When ``True`` keypoint coordinates are stored in each :class:`_LazyYoloSample` (pose path).
            Mutually exclusive with ``include_polygons``; raises :class:`ValueError` when both are ``True``.
        keypoint_schema: Keypoint schema describing class names, per-class keypoint counts, OKS sigmas, keypoint names,
            flip index, and keypoint dimensionality.  When ``None`` and ``include_keypoints=True`` the schema is
            auto-inferred from ``data_file`` via :func:`infer_yolo_keypoint_schema`.

    Returns:
        A ``(classes, samples)`` tuple where ``classes`` is the ordered list of class names and ``samples`` is a list of
        :class:`_LazyYoloSample` objects.

    Examples:
        >>> # Used internally by _build_lazy_yolo_detection_dataset and
        >>> # _build_lazy_yolo_segmentation_dataset — not part of the public API.
        >>> pass
    """
    if include_polygons and include_keypoints:
        raise ValueError("YOLO segmentation masks and keypoints cannot be loaded at the same time.")
    if include_keypoints:
        keypoint_schema = keypoint_schema or infer_yolo_keypoint_schema(data_file)
        classes = keypoint_schema.class_names
        num_keypoints = max(keypoint_schema.num_keypoints_per_class, default=0)
        keypoint_dim = keypoint_schema.keypoint_dim
    else:
        classes = _extract_yolo_class_names(data_file)
        num_keypoints = 0
        keypoint_dim = 0
    samples: list[_LazyYoloSample] = []

    for image_path in _list_yolo_image_paths(img_folder):
        label_path = Path(lb_folder) / f"{Path(image_path).stem}.txt"
        with Image.open(image_path) as image:
            width, height = image.size

        xyxy: list[np.ndarray] = []
        class_id: list[int] = []
        polygons: list[np.ndarray] = []
        keypoints: list[np.ndarray] = []
        if label_path.exists():
            with label_path.open(encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]
            for i, line in enumerate(lines):
                values = line.split()
                if include_keypoints:
                    cid, xyxy_px, keypoints_px = _parse_yolo_pose_label_line(
                        values,
                        i + 1,
                        label_path,
                        len(classes),
                        width,
                        height,
                        num_keypoints=num_keypoints,
                        keypoint_dim=keypoint_dim,
                    )
                    polygon_px = None
                    keypoints.append(keypoints_px)
                else:
                    cid, xyxy_px, polygon_px = _parse_yolo_label_line(
                        values,
                        i + 1,
                        label_path,
                        len(classes),
                        width,
                        height,
                        parse_polygons=include_polygons,
                    )
                class_id.append(cid)
                xyxy.append(xyxy_px)
                if include_polygons and polygon_px is not None:
                    polygons.append(polygon_px)

        samples.append(
            _LazyYoloSample(
                image_path=image_path,
                width=width,
                height=height,
                xyxy=np.array(xyxy, dtype=np.float32).reshape(-1, 4),
                class_id=np.array(class_id, dtype=np.int64),
                polygons=tuple(polygons),
                keypoints=(
                    np.stack(keypoints).astype(np.float32, copy=False)
                    if keypoints
                    else np.zeros((0, num_keypoints, 3), dtype=np.float32)
                ),
            )
        )

    return classes, samples


def _build_lazy_yolo_detection_dataset(img_folder: str, lb_folder: str, data_file: str) -> _LazyYoloDetectionDataset:
    """Build a YOLO detection dataset that stores bounding boxes lazily.

    Unlike :func:`_build_lazy_yolo_segmentation_dataset`, this function does not store polygon coordinates or dense
    masks — only ``xyxy`` boxes are retained, keeping peak memory proportional to the number of annotations.

    Images without a matching ``.txt`` label file are included as *background* samples with empty detections, so
    datasets that mix labelled and unlabelled images are handled correctly.

    Args:
        img_folder: Path to the directory containing images.
        lb_folder: Path to the directory containing YOLO ``.txt`` label files.
        data_file: Path to the ``data.yaml`` / ``data.yml`` file with class names.

    Returns:
        A :class:`_LazyYoloDetectionDataset` whose ``__getitem__`` loads pixel data on demand and returns
        ``sv.Detections`` without mask information.
    """
    classes, samples = _build_yolo_samples(img_folder, lb_folder, data_file, include_polygons=False)
    return _LazyYoloDetectionDataset(classes=classes, samples=samples)


def _build_lazy_yolo_segmentation_dataset(img_folder: str, lb_folder: str, data_file: str) -> _LazyYoloDetectionDataset:
    """Build a YOLO dataset that stores polygons and rasterizes masks on demand.

    Args:
        img_folder: Path to the directory containing images.
        lb_folder: Path to the directory containing YOLO ``.txt`` label files.
        data_file: Path to the ``data.yaml`` / ``data.yml`` file with class names.

    Returns:
        A :class:`_LazyYoloDetectionDataset` whose ``__getitem__`` loads pixel data on demand and rasterizes polygon
        masks into dense boolean tensors.
    """
    classes, samples = _build_yolo_samples(img_folder, lb_folder, data_file, include_polygons=True)
    return _LazyYoloDetectionDataset(classes=classes, samples=samples)


def _build_lazy_yolo_keypoint_dataset(
    img_folder: str,
    lb_folder: str,
    data_file: str,
    keypoint_schema: YoloKeypointSchema,
) -> _LazyYoloDetectionDataset:
    """Build a YOLO pose dataset that stores keypoints without dense masks."""
    classes, samples = _build_yolo_samples(
        img_folder,
        lb_folder,
        data_file,
        include_polygons=False,
        include_keypoints=True,
        keypoint_schema=keypoint_schema,
    )
    return _LazyYoloDetectionDataset(classes=classes, samples=samples)


def _build_coco_api_from_samples(
    classes: list[str],
    dataset: Any,
    keypoint_schema: YoloKeypointSchema | None = None,
) -> Any:
    """Build an in-memory ``pycocotools.COCO`` object from YOLO lazy samples.

    Args:
        classes: Ordered class names where index is the YOLO class ID.
        dataset: Lazy YOLO backend exposing ``__len__`` and either ``get_image_info(idx)`` or ``__getitem__(idx)``.

    Returns:
        Initialized ``pycocotools.COCO`` object with ``dataset`` and indexes.
    """
    from pycocotools.coco import COCO

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    for idx, class_name in enumerate(classes):
        category = {"id": idx, "name": class_name, "supercategory": "none"}
        if keypoint_schema is not None:
            category["keypoints"] = list(keypoint_schema.keypoint_names)
            category["skeleton"] = []
        categories.append(category)

    use_lazy_path = hasattr(dataset, "get_image_info")
    ann_id = 0
    for img_id in range(len(dataset)):
        if use_lazy_path:
            sample = dataset.get_image_info(img_id)
            image_path = sample.image_path
            height, width = sample.height, sample.width
            xyxy = sample.xyxy
            class_id = sample.class_id
            has_masks = len(sample.polygons) > 0
            keypoints = sample.keypoints
        else:
            image_path, image_array, detections = dataset[img_id]
            height, width = image_array.shape[:2]
            xyxy = detections.xyxy
            class_id = detections.class_id
            has_masks = detections.mask is not None
            keypoints = np.zeros((len(xyxy), 0, 3), dtype=np.float32)

        images.append({"id": img_id, "file_name": str(image_path), "height": int(height), "width": int(width)})

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            bbox_x, bbox_y = float(x1), float(y1)
            bbox_w, bbox_h = float(x2 - x1), float(y2 - y1)
            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(class_id[i]),
                "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                "area": float(bbox_w * bbox_h),
                "iscrowd": 0,
            }
            if has_masks:
                # Keep bbox evaluation compatible without eager mask encoding at init.
                ann["segmentation"] = []
            if keypoint_schema is not None:
                keypoints_i = keypoints[i] if i < len(keypoints) else np.zeros((0, 3), dtype=np.float32)
                ann["keypoints"] = keypoints_i.reshape(-1).astype(float).tolist()
                ann["num_keypoints"] = int(np.count_nonzero(keypoints_i[:, 2] > 0))
            annotations.append(ann)
            ann_id += 1

    coco_dataset = {
        "info": {"description": "RF-DETR YOLO dataset"},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    coco = COCO()
    coco.dataset = coco_dataset
    coco.createIndex()
    return coco


def is_valid_yolo_dataset(dataset_dir: str) -> bool:
    """Checks if the specified dataset directory is in yolo format.

    We accept a dataset to be in yolo format if the following conditions are met:
    - The dataset_dir contains a data.yaml or data.yml file
    - The dataset_dir contains "train" and either "valid" or "val" subdirectories,
      each containing "images" and "labels" subdirectories
    - The "test" subdirectory is optional

    .. note::
        This is a coarse filesystem pre-check and intentionally does **not**
        consult ``data.yaml`` split path keys.  Actual split directories are
        resolved by :func:`_resolve_yolo_split_dirs`, which reads the YAML
        first and falls back to the filesystem convention.  When both ``valid/``
        and ``val/`` exist but YAML declares the non-priority one, the two
        functions may pick different directories — this is by design; the
        validity gate is a cheap early filter only.  When ``valid/`` exists,
        it takes precedence over ``val/``.

    Returns:
        ``True`` if the directory satisfies all YOLO format requirements,
        ``False`` otherwise.
    """
    contains_required_yolo_yaml = any(
        os.path.exists(os.path.join(dataset_dir, yaml_file)) for yaml_file in REQUIRED_YOLO_YAML_FILES
    )
    has_train = os.path.exists(os.path.join(dataset_dir, "train"))
    has_val = any(os.path.exists(os.path.join(dataset_dir, d)) for d in _VALID_VAL_DIR_NAMES)
    contains_required_split_dirs = has_train and has_val

    val_dir_name = next(
        (d for d in _VALID_VAL_DIR_NAMES if os.path.exists(os.path.join(dataset_dir, d))),
        "valid",
    )
    active_splits = ["train", val_dir_name]
    contains_required_data_subdirs = all(
        os.path.exists(os.path.join(dataset_dir, split_dir, data_subdir))
        for split_dir in active_splits
        for data_subdir in REQUIRED_DATA_SUBDIRS
    )
    return contains_required_yolo_yaml and contains_required_split_dirs and contains_required_data_subdirs


def _parse_yaml_split_dirs(root: Path, data_file: Path, split: str) -> tuple[Path, Path] | None:
    """Parse ``data_file`` and resolve image/label dirs for ``split``.

    Returns ``None`` when the YAML declares no usable path for the requested
    split, the resolved path does not exist on disk, or a path traversal is
    detected.

    Raises:
        OSError: File I/O failure reading ``data_file``.
        ValueError: Unexpected value type inside the YAML mapping.
        TypeError: Unexpected type inside the YAML mapping.
        yaml.YAMLError: Malformed YAML content.

    Args:
        root: Dataset root directory.
        data_file: Path to ``data.yaml`` or ``data.yml``.
        split: One of ``"train"``, ``"val"``, or ``"test"``.

    Returns:
        ``(images_dir, labels_dir)`` on success, or ``None`` to signal fallback.
    """
    data = _load_yaml_mapping(data_file)
    yaml_base = Path(data.get("path", "")) if data.get("path") else root
    if not yaml_base.is_absolute():
        yaml_base = root / yaml_base

    raw_path: str | None = data.get(split)
    if raw_path is None and split == "val":
        raw_path = data.get("valid")

    if raw_path is None:
        return None

    split_images = yaml_base / raw_path
    # Path traversal guard: reject yaml-declared paths that escape
    # the dataset root (e.g. "../../other_project/val/images").
    try:
        split_images.resolve().relative_to(root.resolve())
    except ValueError:
        return None  # traversal detected; signal fallback

    parts = split_images.parts
    is_dir = split_images.is_dir()
    if is_dir and "images" in parts:
        idx = parts.index("images")
        split_labels = Path(*parts[:idx], "labels", *parts[idx + 1 :])
        if split_labels.is_dir():
            return split_images, split_labels
    elif is_dir:
        sub_images = split_images / "images"
        sub_labels = split_images / "labels"
        if sub_images.is_dir() and sub_labels.is_dir():
            return sub_images, sub_labels
    return None


def _resolve_split_from_yaml(root: Path, data_file: Path, split: str) -> tuple[Path, Path] | None:
    """Try to resolve image and label dirs for ``split`` from ``data_file``.

    Returns ``None`` when the YAML file is absent, declares no usable path for
    the requested split, or the resolved path does not exist on disk.

    Args:
        root: Dataset root directory.
        data_file: Path to ``data.yaml`` or ``data.yml``.
        split: One of ``"train"``, ``"val"``, or ``"test"``.

    Returns:
        ``(images_dir, labels_dir)`` on success, or ``None`` to signal fallback.
    """
    if not data_file.exists():
        return None
    try:
        return _parse_yaml_split_dirs(root, data_file, split)
    except OSError:
        pass
    except (ValueError, TypeError) as exc:
        logger.warning(
            "Could not resolve YAML split path for %r in %s: %s — falling back to Roboflow directory convention.",
            split,
            data_file,
            exc,
        )
    except yaml.YAMLError as exc:
        logger.warning(
            "Failed to parse YAML file %s: %s — falling back to Roboflow directory convention.",
            data_file,
            exc,
        )
    return None


def _resolve_yolo_split_dirs(root: Path, data_file: Path, split: str) -> tuple[Path, Path]:
    """Resolve image and label directories for a YOLO dataset split.

    Supports both Roboflow (``valid/``) and Ultralytics (``val/``, paths in
    ``data.yaml``) layouts.  When the YAML file declares ``train``/``val``/
    ``test`` path keys the function uses those; otherwise it falls back to the
    existing Roboflow convention with an additional check for ``val/`` when
    ``valid/`` is absent.

    When ``split`` is ``"val"``, the function first queries the YAML for a
    ``"val"`` key and, if absent, retries with a ``"valid"`` key before falling
    back to the filesystem convention.

    Labels are derived from the resolved images path by replacing the
    ``"images"`` segment with ``"labels"`` anywhere in the path, mirroring the
    Ultralytics ``img2label_paths`` convention.  This handles both
    trailing-images (``val/images``) and intermediate-images
    (``images/val2017``) layouts.

    Args:
        root: Dataset root directory.
        data_file: Path to ``data.yaml`` or ``data.yml``.
        split: One of ``"train"``, ``"val"``, or ``"test"``.

    Returns:
        ``(images_dir, labels_dir)`` as resolved :class:`~pathlib.Path` objects.
    """
    result = _resolve_split_from_yaml(root, data_file, split)
    if result is not None:
        return result

    roboflow_map = {"train": "train", "val": "valid", "test": "test"}
    mapped = roboflow_map.get(split, split)
    img_dir = root / mapped / "images"
    lb_dir = root / mapped / "labels"

    if split == "val" and not img_dir.exists():
        for alt in _VALID_VAL_DIR_NAMES:
            candidate = root / alt / "images"
            candidate_labels = root / alt / "labels"
            if candidate.is_dir() and candidate_labels.is_dir():
                return candidate, candidate_labels

    return img_dir, lb_dir


class ConvertYolo:
    """Converts supervision Detections to the target dict format expected by RF-DETR.

    Args:
        include_masks: whether to include segmentation masks.
        include_keypoints: whether to include pose keypoints.
        num_keypoints: Number of keypoints per instance when keypoints are enabled.

    Examples:
        >>> import numpy as np
        >>> from supervision import Detections
        >>> from PIL import Image
        >>> # Create a sample image and target
        >>> image = Image.new("RGB", (100, 100))
        >>> detections = Detections(
        ...     xyxy=np.array([[10, 20, 30, 40]]),
        ...     class_id=np.array([0])
        ... )
        >>> target = {"image_id": 0, "detections": detections}
        >>> # Create converter
        >>> converter = ConvertYolo(include_masks=False)
        >>> # Call converter
        >>> img, result = converter(image, target)
        >>> sorted(result.keys())
        ['area', 'boxes', 'image_id', 'iscrowd', 'labels', 'orig_size', 'size']
        >>> result["boxes"].shape
        torch.Size([1, 4])
        >>> result["labels"].tolist()
        [0]
        >>> result["image_id"].tolist()
        [0]
    """

    def __init__(self, include_masks: bool = False, include_keypoints: bool = False, num_keypoints: int = 0):
        self.include_masks = include_masks
        self.include_keypoints = include_keypoints
        self.num_keypoints = num_keypoints

    def __call__(self, image: Image.Image, target: dict) -> tuple:
        """Convert image and YOLO detections to RF-DETR format.

        Args:
            image: PIL Image
            target: dict with 'image_id' and 'detections'

        Returns:
            tuple of (image, target_dict)
        """
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        detections = target["detections"]

        if len(detections) > 0:
            boxes = torch.from_numpy(detections.xyxy).to(torch.float32)
            classes = torch.from_numpy(detections.class_id).to(torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            classes = torch.zeros((0,), dtype=torch.int64)

        # clamp and filter
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target_out = {}
        target_out["boxes"] = boxes
        target_out["labels"] = classes
        target_out["image_id"] = image_id

        # compute area after clamp
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target_out["area"] = area

        iscrowd = torch.zeros((classes.shape[0],), dtype=torch.int64)
        target_out["iscrowd"] = iscrowd

        if self.include_masks:
            if detections.mask is not None and np.size(detections.mask) > 0:
                masks = torch.from_numpy(detections.mask[keep.cpu().numpy()]).to(torch.uint8)
                target_out["masks"] = masks
            else:
                target_out["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)

            target_out["masks"] = target_out["masks"].bool()

        if self.include_keypoints:
            raw_keypoints = target.get("keypoints")
            if raw_keypoints is None:
                # Allocate with pre-filter size so `keep` indexing below is valid
                keypoints = torch.zeros((keep.shape[0], self.num_keypoints, 3), dtype=torch.float32)
            else:
                keypoints = torch.as_tensor(raw_keypoints, dtype=torch.float32).reshape(-1, self.num_keypoints, 3)
            target_out["keypoints"] = keypoints[keep]

        target_out["orig_size"] = torch.as_tensor([int(h), int(w)])
        target_out["size"] = torch.as_tensor([int(h), int(w)])

        return image, target_out


class YoloDetection(VisionDataset):
    """YOLO format dataset with lazy image loading and optional mask support.

    Both detection (``include_masks=False``) and segmentation (``include_masks=True``) paths use a lazy backend: image
    pixels are loaded on demand inside ``__getitem__`` rather than at construction time, which keeps peak RAM
    proportional to the number of annotations rather than to ``N × H × W``.

    Images without a matching ``.txt`` label file are treated as *background* images and produce empty detections.  This
    ensures that datasets containing a mix of annotated and unannotated images are handled correctly in both single-GPU
    and multi-GPU training.

    This class provides a VisionDataset interface compatible with RF-DETR training, matching the API of CocoDetection.

    Args:
        img_folder: Path to the directory containing images
        lb_folder: Path to the directory containing YOLO annotation .txt files
        data_file: Path to data.yaml file containing class names and dataset info
        transforms: Optional transforms to apply to images and targets
        include_masks: Whether to load segmentation masks (for YOLO segmentation format).
            When True polygons are parsed and rasterized on demand; when False only bounding-box coordinates are stored.
        include_keypoints: Whether to load Ultralytics YOLO pose keypoints.
        num_keypoints_per_class: Optional keypoint schema used by RF-DETR.
    """

    def __init__(
        self,
        img_folder: str,
        lb_folder: str,
        data_file: str,
        transforms=None,
        include_masks: bool = False,
        include_keypoints: bool = False,
        num_keypoints_per_class: list[int] | None = None,
    ):
        if include_masks and include_keypoints:
            raise ValueError("YOLO segmentation masks and keypoints cannot be loaded at the same time.")
        super().__init__(img_folder)
        self._transforms = transforms
        self.include_masks = include_masks
        self.include_keypoints = include_keypoints
        if include_keypoints:
            try:
                self.keypoint_schema = infer_yolo_keypoint_schema(data_file)
            except (FileNotFoundError, ValueError, OSError) as exc:
                raise ValueError(f"YOLO keypoint training requires kpt_shape metadata in {data_file!r}.") from exc
        else:
            self.keypoint_schema = None
        self.num_keypoints = max(num_keypoints_per_class or [], default=0)
        if self.keypoint_schema is not None:
            self.num_keypoints = max(self.keypoint_schema.num_keypoints_per_class, default=self.num_keypoints)
        self.prepare = ConvertYolo(
            include_masks=include_masks,
            include_keypoints=include_keypoints,
            num_keypoints=self.num_keypoints,
        )
        if include_keypoints:
            self.sv_dataset = _build_lazy_yolo_keypoint_dataset(
                img_folder,
                lb_folder,
                data_file,
                self.keypoint_schema,
            )
        elif include_masks:
            self.sv_dataset = _build_lazy_yolo_segmentation_dataset(img_folder, lb_folder, data_file)
        else:
            self.sv_dataset = _build_lazy_yolo_detection_dataset(img_folder, lb_folder, data_file)

        self.classes = self.sv_dataset.classes
        self.ids = list(range(len(self.sv_dataset)))

        # Create COCO-compatible API for evaluation
        self.coco = _build_coco_api_from_samples(self.classes, self.sv_dataset, self.keypoint_schema)

    def __len__(self) -> int:
        return len(self.sv_dataset)

    def __getitem__(self, idx: int):
        image_id = self.ids[idx]
        image_path, rgb_image, detections = self.sv_dataset[idx]

        img = Image.fromarray(rgb_image)

        target = {"image_id": image_id, "detections": detections}
        if self.include_keypoints:
            target["keypoints"] = self.sv_dataset.get_image_info(idx).keypoints
        img, target = self.prepare(img, target)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target


def build_roboflow_from_yolo(image_set: str, args: Any, resolution: int) -> YoloDetection:
    """Build a YOLO-format dataset from Roboflow or Ultralytics directory layouts.

    Supports both Roboflow (``train/valid/test``) and Ultralytics (``train/val/test``,
    with ``data.yaml`` path keys) directory structures.  Split directories are resolved
    via :func:`_resolve_yolo_split_dirs`.

    Args:
        image_set: Dataset split to load. One of ``"train"``, ``"val"``, or
            ``"test"``.
        args: Argument namespace. The following attributes are consumed:
            ``dataset_dir``, ``square_resize_div_64``, ``aug_config``, ``segmentation_head``, ``multi_scale``,
            ``expanded_scales``, ``do_random_resize_via_padding``, ``patch_size``, ``num_windows``. ``aug_config`` is
            forwarded to the transform builder; when ``None`` the builder falls back to the default
            :data:`~rfdetr.datasets.aug_configs.AUG_CONFIG`.
        resolution: Target square resolution in pixels.

    Returns:
        A :class:`YoloDetection` dataset instance ready for use with a DataLoader.
    """
    root = Path(args.dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"YOLO dataset root not found: {root}")

    # Prefer data.yaml; fall back to data.yml if present; default to data.yaml for error reporting
    data_file = next((root / f for f in REQUIRED_YOLO_YAML_FILES if (root / f).exists()), root / "data.yaml")
    split_key = image_set.split("_")[0]
    img_folder, lb_folder = _resolve_yolo_split_dirs(root, data_file, split_key)
    square_resize_div_64 = getattr(args, "square_resize_div_64", False)
    include_masks = getattr(args, "segmentation_head", False)
    multi_scale = getattr(args, "multi_scale", False)
    expanded_scales = getattr(args, "expanded_scales", False)
    do_random_resize_via_padding = getattr(args, "do_random_resize_via_padding", False)
    patch_size = getattr(args, "patch_size", 16)
    num_windows = getattr(args, "num_windows", 4)
    aug_config = getattr(args, "aug_config", None)
    include_keypoints = getattr(args, "use_grouppose_keypoints", False)
    num_keypoints_per_class = getattr(args, "num_keypoints_per_class", [])
    keypoint_flip_pairs: list[int] | None = (
        (getattr(args, "keypoint_flip_pairs", []) or []) if include_keypoints else None
    )
    resolved_augmentation_backend = _resolve_runtime_augmentation_backend(getattr(args, "augmentation_backend", "cpu"))
    gpu_postprocess = resolved_augmentation_backend != "cpu"

    if include_keypoints:
        try:
            infer_yolo_keypoint_schema(data_file)
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise ValueError(
                "YOLO keypoint training requires an Ultralytics pose data.yaml/data.yml with valid kpt_shape metadata."
            ) from exc

    if square_resize_div_64:
        dataset = YoloDetection(
            img_folder=str(img_folder),
            lb_folder=str(lb_folder),
            data_file=str(data_file),
            transforms=make_coco_transforms_square_div_64(
                image_set,
                resolution,
                multi_scale=multi_scale,
                expanded_scales=expanded_scales,
                skip_random_resize=not do_random_resize_via_padding,
                patch_size=patch_size,
                num_windows=num_windows,
                aug_config=aug_config,
                gpu_postprocess=gpu_postprocess,
                keypoint_flip_pairs=keypoint_flip_pairs,
            ),
            include_masks=include_masks,
            include_keypoints=include_keypoints,
            num_keypoints_per_class=num_keypoints_per_class,
        )
    else:
        dataset = YoloDetection(
            img_folder=str(img_folder),
            lb_folder=str(lb_folder),
            data_file=str(data_file),
            transforms=make_coco_transforms(
                image_set,
                resolution,
                multi_scale=multi_scale,
                expanded_scales=expanded_scales,
                skip_random_resize=not do_random_resize_via_padding,
                patch_size=patch_size,
                num_windows=num_windows,
                aug_config=aug_config,
                gpu_postprocess=gpu_postprocess,
                keypoint_flip_pairs=keypoint_flip_pairs,
            ),
            include_masks=include_masks,
            include_keypoints=include_keypoints,
            num_keypoints_per_class=num_keypoints_per_class,
        )
    return dataset

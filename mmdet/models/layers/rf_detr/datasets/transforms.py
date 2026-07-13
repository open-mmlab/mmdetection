# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""Transforms and data augmentation for both image + bbox."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from functools import cache
from typing import Any

try:
    import albumentations as alb
except ImportError:
    alb = None  # type: ignore[assignment]
import numpy as np
import PIL
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import Normalize as _TVNormalize

from mmdet.models.layers.rf_detr.datasets._aug_utils import filter_keypoint_hflip_augmentations
from mmdet.models.layers.rf_detr.utilities.box_ops import box_xyxy_to_cxcywh
from mmdet.models.layers.rf_detr.utilities.logger import get_logger

logger = get_logger()


class Normalize:
    def __init__(
        self,
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        std: tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        self._normalize = _TVNormalize(mean, std)

    def __call__(self, image: Tensor, target: dict[str, Any] | None = None) -> tuple[Tensor, dict[str, Any] | None]:
        """Normalize image and convert target coordinates to relative format.

        Applies ImageNet-style channel normalization to the image, then converts
        bounding boxes from absolute xyxy pixel coordinates to normalized cxcywh
        format (divided by ``[w, h, w, h]``) and scales keypoint x/y by image
        width/height respectively.

        Args:
            image: CHW float tensor to normalize.
            target: Optional dict with keys ``"boxes"`` (xyxy pixel coords,
                shape ``[N, 4]``) and/or ``"keypoints"`` (shape ``[N, K, 3]``
                where the third channel is visibility). Mutated copy returned;
                original is not modified.

        Returns:
            Tuple of ``(normalized_image, target)`` where ``target`` has boxes
            in normalized cxcywh format and keypoints scaled to ``[0, 1]``, or
            ``(normalized_image, None)`` when ``target`` is ``None``.

        Examples:
            >>> import torch
            >>> normalize = Normalize()
            >>> img = torch.zeros(3, 64, 64)
            >>> out_img, out_tgt = normalize(img, None)
            >>> out_tgt is None
            True
        """
        image = self._normalize(image)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        if "keypoints" in target:
            keypoints = target["keypoints"].clone()  # shape: (N, K, 3) — x, y, visibility
            keypoints[..., 0] = keypoints[..., 0] / w
            keypoints[..., 1] = keypoints[..., 1] / h
            target["keypoints"] = keypoints
        return image, target


# Albumentations wrapper for RF-DETR

# Geometric transforms that affect bounding boxes
# These transforms modify spatial coordinates, so bounding boxes must be transformed accordingly.
# For custom geometric transforms, add the class name to this set.
GEOMETRIC_TRANSFORMS = {
    # Flips and transpositions
    "HorizontalFlip",
    "VerticalFlip",
    "Flip",
    "Transpose",
    "D4",
    # Rotations and affine transforms
    "Rotate",
    "RandomRotate90",
    "Affine",
    "ShiftScaleRotate",
    "SafeRotate",
    # Crops
    "RandomCrop",
    "RandomSizedCrop",
    "CenterCrop",
    "Crop",
    "CropNonEmptyMaskIfExists",
    "RandomCropNearBBox",
    "RandomCropFromBorders",
    "RandomSizedBBoxSafeCrop",
    "BBoxSafeRandomCrop",
    "AtLeastOneBBoxRandomCrop",
    "RandomResizedCrop",
    "CropAndPad",
    # Perspective and distortions
    "Perspective",
    "ElasticTransform",
    "GridDistortion",
    "GridElasticDeform",
    "OpticalDistortion",
    "PiecewiseAffine",
    "ThinPlateSpline",
    "RandomGridShuffle",
    # Resize operations
    "Resize",
    "SmallestMaxSize",
    "LongestMaxSize",
    "RandomScale",
    "Downscale",
    # Padding and symmetry
    "PadIfNeeded",
    "Pad",
    "SquareSymmetry",
}

# Albumentations container/meta transforms that hold nested transforms
ALBUMENTATIONS_CONTAINERS = frozenset({"OneOf", "SomeOf", "Sequential"})


def _is_geometric_transform(transform: alb.BasicTransform) -> bool:
    """Return True if transform (or any nested transform) affects spatial coordinates.

    For container transforms such as ``A.OneOf`` or ``A.Sequential``, returns ``True`` when *any* nested transform is
    geometric so that bounding-box handling is enabled for the whole container.

    Args:
        transform: Albumentations transform to inspect.

    Returns:
        ``True`` if the transform modifies spatial layout; ``False`` otherwise.

    Examples:
        >>> from albumentations import GaussianBlur, HorizontalFlip, OneOf
        >>> _is_geometric_transform(HorizontalFlip())
        True
        >>> _is_geometric_transform(GaussianBlur())
        False
        >>> _is_geometric_transform(OneOf([HorizontalFlip(), GaussianBlur()]))
        True
    """
    if type(transform).__name__ in GEOMETRIC_TRANSFORMS:
        return True
    # Recursively check nested transforms in container transforms
    if hasattr(transform, "transforms"):
        return any(_is_geometric_transform(t) for t in transform.transforms)
    return False


def _build_albu_transform(name: str, params: dict[str, Any]) -> alb.BasicTransform:
    """Build a single Albumentations transform from its name and parameter dict.

    Handles container transforms (``OneOf``, ``SomeOf``, ``Sequential``) by recursively building the nested
    ``transforms`` list.  Leaf transforms are instantiated directly from the ``albumentations`` namespace.

    Both ``OneOf`` and ``Sequential`` always fire (``p=1.0`` is forced, ignoring any user-supplied ``p``).  For
    ``OneOf``, which child is applied is determined by the children's own ``p`` values; at least one nested transform is
    required.  ``Sequential`` runs all transforms in order.

    Args:
        name: Transform name (e.g. ``"HorizontalFlip"``, ``"OneOf"``).
        params: Parameter dictionary for the transform.  For container transforms
            the dict must contain a ``"transforms"`` key whose value is a list of single-key dicts ``{name: params}``.

    Returns:
        Instantiated Albumentations transform.

    Raises:
        ImportError: If Albumentations is not installed.
        ValueError: If ``name`` is unknown or ``params`` is malformed.

    Examples:
        >>> from albumentations import HorizontalFlip, OneOf
        >>> t = _build_albu_transform("HorizontalFlip", {"p": 0.5})
        >>> isinstance(t, HorizontalFlip)
        True
        >>> container = _build_albu_transform(
        ...     "OneOf",
        ...     {"transforms": [{"HorizontalFlip": {"p": 1.0}}, {"VerticalFlip": {"p": 1.0}}]},
        ... )
        >>> isinstance(container, OneOf)
        True
    """
    if alb is None:
        raise ImportError(
            "Albumentations is required to build RF-DETR dataset transforms. "
            "Install the project dependencies with `uv sync --all-groups` or install albumentations."
        )

    if name in ALBUMENTATIONS_CONTAINERS:
        raw_nested = params.get("transforms", [])
        if not isinstance(raw_nested, list):
            raise ValueError(f"'{name}.transforms' must be a list, got {type(raw_nested).__name__}")
        nested_transforms: list[alb.BasicTransform] = []
        for entry in raw_nested:
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError(f"Each nested transform entry must be a single-key dict, got {entry!r}")
            nested_name, nested_params = next(iter(entry.items()))
            if not isinstance(nested_params, dict):
                raise ValueError(
                    f"Parameters for nested transform '{nested_name}' must be a dict, "
                    f"got {type(nested_params).__name__}"
                )
            nested_transforms.append(_build_albu_transform(nested_name, nested_params))

        if name == "OneOf":
            if not nested_transforms:
                raise ValueError("'OneOf' requires at least one transform")
            other_params = {k: v for k, v in params.items() if k not in ("transforms", "p")}
            other_params["p"] = 1.0  # OneOf always fires; selection is via per-child p
        elif name == "Sequential":
            other_params = {k: v for k, v in params.items() if k not in ("transforms", "p")}
            other_params["p"] = 1.0  # Sequential always runs all transforms
        else:
            other_params = {k: v for k, v in params.items() if k != "transforms"}

        container_cls = getattr(alb, name, None)
        if container_cls is None:
            raise ValueError(f"Unknown Albumentations container: {name!r}")
        return container_cls(transforms=nested_transforms, **other_params)

    aug_cls = getattr(alb, name, None)
    if aug_cls is None:
        raise ValueError(f"Unknown Albumentations transform: {name!r}")
    return aug_cls(**_normalize_albu_params(name, params, aug_cls))


@cache
def _random_sized_crop_uses_size_param(aug_cls: type) -> bool:
    """Return whether ``RandomSizedCrop`` expects a ``size`` keyword.

    The Albumentations 2.x API changed ``RandomSizedCrop`` from separate ``height``/``width`` parameters to a single
    ``size=(height, width)`` parameter. This helper caches the signature check per class so repeated transform
    construction during dataset setup does not repeat introspection.

    Args:
        aug_cls: Albumentations transform class to inspect.

    Returns:
        ``True`` when the class accepts a ``size`` keyword argument; otherwise ``False``.
    """
    signature = inspect.signature(aug_cls.__init__)
    return "size" in signature.parameters


def _normalize_albu_params(name: str, params: dict[str, Any], aug_cls: type) -> dict[str, Any]:
    """Normalize transform params across Albumentations API variations.

    Currently this adapts ``RandomSizedCrop`` arguments so a config using ``height``/``width`` works on Albumentations
    2.x and a config using ``size=(height, width)`` still works on Albumentations 1.x.

    Args:
        name: Albumentations transform name.
        params: Raw transform parameter mapping from config.
        aug_cls: Albumentations transform class that will be instantiated.

    Returns:
        A normalized copy of ``params`` suitable for the installed Albumentations version.

    Examples:
        >>> class CropV2:
        ...     def __init__(self, *, size, min_max_height): ...
        >>> _normalize_albu_params(
        ...     "RandomSizedCrop",
        ...     {"min_max_height": [384, 600], "height": 640, "width": 640},
        ...     CropV2,
        ... )
        {'min_max_height': [384, 600], 'size': (640, 640)}
    """
    normalized_params = dict(params)
    if name != "RandomSizedCrop":
        return normalized_params

    uses_size = _random_sized_crop_uses_size_param(aug_cls)

    if uses_size:
        # Albumentations 2.x-style API: expects ``size`` and does not accept
        # separate ``height``/``width`` kwargs.
        has_size = "size" in normalized_params
        has_height = "height" in normalized_params
        has_width = "width" in normalized_params

        if has_size:
            # When ``size`` is already provided, drop any legacy ``height``/``width``
            # so they are never forwarded as unexpected keyword arguments.
            normalized_params.pop("height", None)
            normalized_params.pop("width", None)
            return normalized_params

        if has_height and has_width:
            height = normalized_params.pop("height")
            width = normalized_params.pop("width")
            normalized_params["size"] = (height, width)
            return normalized_params

        if has_height != has_width:
            # One of ``height``/``width`` was provided without the other and no
            # explicit ``size`` was given. This is ambiguous for the
            # Albumentations 2.x API, so raise a targeted error instead of
            # silently dropping parameters and surfacing a generic
            # "missing required argument 'size'" error later.
            missing = "width" if has_height and not has_width else "height"
            raise ValueError(
                f"RandomSizedCrop for the installed Albumentations version expects "
                f"'size=(height, width)'. Received only one of 'height'/'width' "
                f"without 'size' (missing '{missing}')."
            )

        # No ``size``, ``height``, or ``width`` provided; let Albumentations
        # surface its own error about missing required arguments.
        return normalized_params

    # NOTE: For Albumentations builds >=1.4.24, ``RandomSizedCrop`` typically
    # uses a ``size`` argument and ``uses_size`` will be True. This project
    # also supports Albumentations 1.x-style APIs (and synthetic v1-style
    # classes used in tests) where ``RandomSizedCrop`` may not accept ``size``
    # directly; in those cases we map a provided ``size`` tuple back to
    # separate ``height``/``width`` kwargs for compatibility.
    if not uses_size and "size" in normalized_params:  # v1-style API compatibility path
        size = normalized_params.get("size")
        if isinstance(size, Sequence) and len(size) == 2:
            normalized_params.setdefault("height", size[0])
            normalized_params.setdefault("width", size[1])
            # Only remove ``size`` after a successful conversion; otherwise leave
            # it so Albumentations can raise an appropriate error.
            normalized_params.pop("size", None)

    return normalized_params


class AlbumentationsWrapper:
    """Wrapper to apply Albumentations transforms to (image, target) tuples.

    This wrapper integrates Albumentations transforms with RF-DETR's data pipeline, automatically handling bounding box
    and segmentation mask transformations for geometric augmentations while preserving the (image, target) tuple format.

    The wrapper automatically detects transform types:
    - **Geometric transforms** (flips, rotations, crops): Bounding boxes and instance
      masks are transformed along with the image to maintain correct object localization.
    - **Pixel-level transforms** (blur, color adjustments, noise): Bounding boxes and
      masks remain unchanged as only pixel values are modified.

    Detection checks the transform class name against ``GEOMETRIC_TRANSFORMS`` and recursively inspects nested container
    transforms (for example ``OneOf`` and ``Sequential``). For geometric transforms, bbox_params are automatically
    configured to handle coordinate transformations, clip boxes to image boundaries, and remove invalid boxes.

    Args:
        transform: Albumentations transform to apply (e.g., alb.HorizontalFlip, alb.GaussianBlur).
        keypoint_flip_pairs: Joint index pairs for left/right swapping after a horizontal flip.
            ``None`` (default) means a detection pipeline -- no keypoint handling.
            An empty list ``[]`` marks a keypoint pipeline without semantic flip
            pairs, so horizontal-flip transforms should have been stripped from
            config before this point.

    Examples:
        >>> from albumentations import GaussianBlur, HorizontalFlip
        >>> # Geometric transform - automatically transforms boxes
        >>> wrapper = AlbumentationsWrapper(HorizontalFlip(p=1.0))
        >>> image = Image.new("RGB", (300, 400))
        >>> target = {"boxes": torch.tensor([[10, 20, 100, 200]]), "labels": torch.tensor([1])}
        >>> aug_image, aug_target = wrapper(image, target)

        >>> # Pixel-level transform - automatically preserves boxes
        >>> wrapper = AlbumentationsWrapper(GaussianBlur(p=1.0))
        >>> aug_image, aug_target = wrapper(image, target)

    Note:
        For custom geometric transforms, add the transform class name to the GEOMETRIC_TRANSFORMS set at module level.
    """

    def __init__(self, transform: alb.BasicTransform, keypoint_flip_pairs: list[int] | None = None) -> None:
        # Auto-detect if transform is geometric (recursively for containers)
        self._is_geometric = _is_geometric_transform(transform)
        self._keypoint_flip_pairs = list(keypoint_flip_pairs or [])

        if self._is_geometric:
            # Wrap geometric transform with bbox handling capabilities
            # bbox_params configure how Albumentations should transform bounding boxes:
            needs_replay = bool(self._keypoint_flip_pairs)
            if needs_replay and not hasattr(alb, "ReplayCompose"):
                logger.warning(
                    "albumentations.ReplayCompose not available; horizontal-flip keypoint "
                    "slot swapping is disabled. Upgrade albumentations to >=1.3."
                )
            compose_cls = alb.ReplayCompose if (needs_replay and hasattr(alb, "ReplayCompose")) else alb.Compose
            self.transform = compose_cls(
                [transform],
                bbox_params=alb.BboxParams(
                    format="pascal_voc",  # Boxes are in (x1, y1, x2, y2) format
                    label_fields=["category_ids", "idxs"],  # Track labels and indices for per-instance field sync
                    min_visibility=0.0,  # Remove boxes with zero visibility/area after transformation
                    clip=True,  # Clip box coordinates to image boundaries after transformation
                ),
                keypoint_params=alb.KeypointParams(
                    format="xy",
                    label_fields=["keypoint_instance_ids", "keypoint_point_ids", "keypoint_visibility"],
                    remove_invisible=False,
                ),
            )
        else:
            # Wrap non-geometric transform without bbox handling
            # Simpler composition since boxes don't need transformation
            self.transform = alb.Compose([transform])

    def __repr__(self) -> str:
        """Return a readable string representation of the wrapper.

        Returns:
            Representation including the wrapped transform and type.
        """
        transform = None
        if isinstance(self.transform, alb.Compose):
            for candidate in self.transform.transforms:
                if isinstance(candidate, alb.BasicTransform):
                    transform = candidate
                    break
        elif isinstance(self.transform, alb.BasicTransform):
            transform = self.transform

        if transform is None:
            return object.__repr__(self)

        transform_type = "geometric" if self._is_geometric else "pixel-level"
        return f"{self.__class__.__name__}(transform={transform}, type={transform_type})"

    @staticmethod
    def _boxes_to_numpy(boxes: Tensor | np.ndarray) -> np.ndarray:
        """Convert boxes to numpy array and validate shape.

        >>> import torch
        >>> boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        >>> AlbumentationsWrapper._boxes_to_numpy(boxes).shape
        (1, 4)
        """
        boxes_np = boxes.cpu().numpy() if torch.is_tensor(boxes) else np.array(boxes)
        if len(boxes_np.shape) != 2 or boxes_np.shape[1] != 4:
            raise ValueError(f"boxes must have shape (N, 4), got {boxes_np.shape}")
        return boxes_np

    @staticmethod
    def _keypoints_to_numpy(keypoints: Tensor | np.ndarray, num_boxes: int) -> np.ndarray:
        """Convert keypoints to numpy array and validate shape.

        >>> import torch
        >>> keypoints = torch.tensor([[[10.0, 20.0, 2.0]]])
        >>> AlbumentationsWrapper._keypoints_to_numpy(keypoints, 1).shape
        (1, 1, 3)
        """
        keypoints_np = keypoints.cpu().numpy() if torch.is_tensor(keypoints) else np.array(keypoints)
        if len(keypoints_np.shape) != 3 or keypoints_np.shape[2] != 3:
            raise ValueError(f"keypoints must have shape (N, K, 3), got {keypoints_np.shape}")
        if keypoints_np.shape[0] != num_boxes:
            raise ValueError(
                f"keypoints first dimension must match number of boxes ({num_boxes}), got {keypoints_np.shape[0]}"
            )
        return keypoints_np

    @staticmethod
    def _build_albu_keypoints(
        keypoints_np: np.ndarray,
        idxs: list[int],
    ) -> dict[str, Any]:
        """Flatten per-instance keypoints into Albumentations keypoint fields.

        >>> keypoints = np.array([[[10.0, 20.0, 2.0], [0.0, 0.0, 0.0]]], dtype=np.float32)
        >>> fields = AlbumentationsWrapper._build_albu_keypoints(keypoints, [0])
        >>> fields["keypoints"]
        [(10.0, 20.0), (0.0, 0.0)]
        """
        albu_keypoints: list[tuple[float, float]] = []
        instance_ids: list[float] = []
        point_ids: list[float] = []
        visibility: list[float] = []
        for original_idx in idxs:
            for point_idx, point in enumerate(keypoints_np[original_idx]):
                x, y, visible = point.tolist()
                albu_keypoints.append((float(x), float(y)))
                instance_ids.append(float(original_idx))
                point_ids.append(float(point_idx))
                visibility.append(float(visible))
        return {
            "keypoints": albu_keypoints,
            "keypoint_instance_ids": instance_ids,
            "keypoint_point_ids": point_ids,
            "keypoint_visibility": visibility,
        }

    @staticmethod
    def _replay_contains_horizontal_flip(replay: Any) -> bool:
        """Return whether Albumentations replay metadata applied a horizontal flip.

        Args:
            replay: ``ReplayCompose`` metadata from an Albumentations call.

        Returns:
            ``True`` only when a horizontal mirror transform was actually applied.
        """
        if not isinstance(replay, dict):
            return False

        transforms = replay.get("transforms")
        if isinstance(transforms, list):
            return any(AlbumentationsWrapper._replay_contains_horizontal_flip(transform) for transform in transforms)

        if not replay.get("applied", False):
            return False

        transform_name = str(replay.get("__class_fullname__", "")).rsplit(".", 1)[-1]
        if transform_name == "HorizontalFlip":
            return True
        if transform_name == "Flip":
            params = replay.get("params") or {}
            return int(params.get("axis", params.get("d", -1))) == 1
        if transform_name in {"D4", "SquareSymmetry"}:
            params = replay.get("params") or {}
            return str(params.get("group_element")) == "h"
        return False

    @staticmethod
    def _rebuild_keypoints_from_albu(
        augmented: dict[str, Any],
        kept_idxs: list[int],
        keypoints_np: np.ndarray,
        flip_pairs: list[int] | None = None,
        did_flip: bool = False,
    ) -> Tensor:
        """Rebuild transformed keypoints and keep them synchronized with kept boxes.

        Args:
            augmented: Augmented output dict from Albumentations.
            kept_idxs: Original instance indices of surviving boxes.
            keypoints_np: Original keypoint array, shape (N_orig, K, 3).
            flip_pairs: Flat list of paired joint indices ``[a0, b0, a1, b1, ...]``
                to swap when a horizontal flip is detected.  Each consecutive pair
                ``(flip_pairs[i], flip_pairs[i+1])`` names two joints that are
                left/right mirrors of each other (e.g., left_eye, right_eye).
            did_flip: Whether a horizontal flip was applied this step.

        Returns:
            Keypoint tensor of shape ``(len(kept_idxs), K, 3)``.
        """
        num_keypoints = keypoints_np.shape[1]
        keypoints_out = np.zeros((len(kept_idxs), num_keypoints, 3), dtype=np.float32)
        kept_position_by_idx = {int(original_idx): position for position, original_idx in enumerate(kept_idxs)}
        height, width = augmented["image"].shape[:2]

        albu_kps = augmented.get("keypoints", [])
        if albu_kps:
            inst_ids = augmented.get("keypoint_instance_ids", [])
            pt_ids = augmented.get("keypoint_point_ids", [])
            visible = augmented.get("keypoint_visibility", [])
            xy = np.asarray([(float(p[0]), float(p[1])) for p in albu_kps], dtype=np.float32)
            inst = np.array([kept_position_by_idx.get(int(ii), -1) for ii in inst_ids], dtype=np.intp)
            ptid = np.array([int(p) for p in pt_ids], dtype=np.intp)
            vis = np.asarray([float(v) for v in visible], dtype=np.float32)
            valid = (
                (inst >= 0)
                & (ptid >= 0)
                & (ptid < num_keypoints)
                & (vis > 0)
                & (xy[:, 0] >= 0)
                & (xy[:, 0] < width)
                & (xy[:, 1] >= 0)
                & (xy[:, 1] < height)
            )
            valid_idx = np.where(valid)[0]
            if len(valid_idx) > 0:
                keypoints_out[inst[valid_idx], ptid[valid_idx]] = np.column_stack([xy[valid_idx], vis[valid_idx]])

        result = torch.as_tensor(keypoints_out, dtype=torch.float32)

        if did_flip and flip_pairs:
            # Build permutation index and apply in a single indexed gather (O(1) dispatch vs O(K) clones)
            num_kpts = result.shape[1]
            perm = torch.arange(num_kpts)
            for i in range(0, len(flip_pairs) - 1, 2):
                ai, bi = flip_pairs[i], flip_pairs[i + 1]
                if ai < num_kpts and bi < num_kpts:
                    perm[ai] = bi
                    perm[bi] = ai
            result = result[:, perm, :]

        return result

    @staticmethod
    def _clear_per_instance_fields(target: dict[str, Any], num_boxes: int) -> dict[str, Any]:
        """Clear all per-instance fields when no boxes remain.

        >>> import torch
        >>> target = {"area": torch.tensor([100, 200]), "iscrowd": torch.tensor([0, 1])}
        >>> cleared = AlbumentationsWrapper._clear_per_instance_fields(target, 2)
        >>> cleared["area"].shape
        torch.Size([0])
        """
        # Fields that are global properties, not per-instance
        global_fields = {"boxes", "labels", "orig_size", "size", "image_id"}

        result = {}
        for key, value in target.items():
            if key in global_fields:
                continue
            if torch.is_tensor(value):
                if value.ndim >= 1 and value.shape[0] == num_boxes:
                    result[key] = value.new_empty((0, *value.shape[1:]))
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                if len(value) == num_boxes:
                    result[key] = []
        return result

    @staticmethod
    def _filter_per_instance_fields(target: dict[str, Any], num_boxes: int, kept_idxs: list[int]) -> dict[str, Any]:
        """Filter per-instance fields to match kept box indices.

        >>> import torch
        >>> target = {"area": torch.tensor([100, 200, 300]), "iscrowd": torch.tensor([0, 0, 1])}
        >>> filtered = AlbumentationsWrapper._filter_per_instance_fields(target, 3, [0, 2])
        >>> filtered["area"].tolist()
        [100, 300]
        """
        # Fields that are global properties, not per-instance
        global_fields = {"boxes", "labels", "orig_size", "size", "image_id"}

        result = {}
        kept_idxs_tensor = torch.as_tensor(kept_idxs, dtype=torch.long)
        for key, value in target.items():
            if key in global_fields:
                continue
            if torch.is_tensor(value):
                if value.ndim >= 1 and value.shape[0] == num_boxes:
                    result[key] = value[kept_idxs_tensor]
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                if len(value) == num_boxes:
                    result[key] = [value[i] for i in kept_idxs]
        return result

    def _apply_geometric_transform(
        self, image_np: np.ndarray, target: dict[str, Any], labels: list[int]
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Apply geometric transform to image with boxes and optionally masks.

        Converts data to Albumentations format, applies the transform, and converts back to RF-DETR format. Handles box
        removal and per-instance field filtering.

        Args:
            image_np: Numpy array of image in HWC format.
            target: Target dictionary with 'boxes' and optionally 'masks'.
            labels: List of category labels.

        Returns:
            Tuple of (transformed PIL Image, transformed target dict).

        >>> import torch
        >>> from albumentations import HorizontalFlip
        >>> wrapper = AlbumentationsWrapper(HorizontalFlip(p=1.0))
        >>> img = np.ones((100, 100, 3), dtype=np.uint8)
        >>> tgt = {"boxes": torch.tensor([[10, 20, 30, 40]]), "labels": torch.tensor([1])}
        >>> img_out, tgt_out = wrapper._apply_geometric_transform(img, tgt, [1])
        >>> tgt_out["boxes"].shape
        torch.Size([1, 4])
        """
        boxes_np = self._boxes_to_numpy(target["boxes"])
        num_boxes = boxes_np.shape[0]
        # Track indices to keep per-instance fields synchronized
        idxs = list(range(num_boxes))
        masks_list = None
        if "masks" in target:
            masks = target["masks"]
            masks_np = masks.cpu().numpy() if torch.is_tensor(masks) else np.array(masks)
            if masks_np.ndim != 3:
                raise ValueError(f"masks must have shape (N, H, W), got {masks_np.shape}")
            masks_np = masks_np.astype(np.uint8, copy=False)
            masks_list = [mask for mask in masks_np]
        keypoints_np = None
        if "keypoints" in target:
            keypoints_np = self._keypoints_to_numpy(target["keypoints"], num_boxes)
        # Filter out degenerate boxes (zero-width or zero-height) before passing to
        # Albumentations. Such boxes arise when an annotation sits exactly on or beyond
        # the image boundary so that x_min == x_max (or y_min == y_max) after clipping.
        # Albumentations' check_bboxes would raise ValueError for these inputs.
        if num_boxes > 0:
            valid_mask = (boxes_np[:, 2] > boxes_np[:, 0]) & (boxes_np[:, 3] > boxes_np[:, 1])
            if not valid_mask.all():
                valid_positions = np.where(valid_mask)[0].tolist()
                boxes_np = boxes_np[valid_mask]
                labels = [labels[i] for i in valid_positions]
                # idxs carries original indices so downstream _filter_per_instance_fields
                # can correctly slice fields from the un-filtered target.
                idxs = [idxs[i] for i in valid_positions]
        # Apply transform
        transform_kwargs = {"image": image_np, "bboxes": boxes_np, "category_ids": labels, "idxs": idxs}
        if masks_list is not None and len(masks_list) > 0:
            transform_kwargs["masks"] = masks_list
        if keypoints_np is not None:
            transform_kwargs.update(self._build_albu_keypoints(keypoints_np, idxs))
        else:
            transform_kwargs.update(
                {
                    "keypoints": [],
                    "keypoint_instance_ids": [],
                    "keypoint_point_ids": [],
                    "keypoint_visibility": [],
                }
            )
        augmented = self.transform(**transform_kwargs)
        target_out: dict[str, Any] = target.copy()
        bboxes_aug = augmented["bboxes"]
        kept_idxs = [int(idx) for idx in augmented.get("idxs", idxs)]
        # Update target with transformed boxes and labels
        if len(bboxes_aug) == 0:
            target_out["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target_out["labels"] = torch.zeros((0,), dtype=torch.long)
            target_out.update(self._clear_per_instance_fields(target, num_boxes))
            # Override masks after _clear_per_instance_fields to ensure bool dtype.
            if "masks" in target:
                aug_height, aug_width = augmented["image"].shape[:2]
                target_out["masks"] = torch.zeros((0, aug_height, aug_width), dtype=torch.bool)
        else:
            target_out["boxes"] = torch.as_tensor(bboxes_aug, dtype=torch.float32).reshape(-1, 4)
            target_out["labels"] = torch.tensor(augmented["category_ids"], dtype=torch.long)
            target_out.update(self._filter_per_instance_fields(target, num_boxes, kept_idxs))
            # Recompute area from the transformed box coordinates so it stays consistent with
            # the new image scale (e.g. after resize the original COCO area values are stale).
            if "area" in target_out:
                boxes = target_out["boxes"]
                target_out["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if keypoints_np is not None:
                did_flip = (
                    self._replay_contains_horizontal_flip(augmented.get("replay"))
                    if self._keypoint_flip_pairs
                    else False
                )
                target_out["keypoints"] = self._rebuild_keypoints_from_albu(
                    augmented,
                    kept_idxs,
                    keypoints_np,
                    flip_pairs=self._keypoint_flip_pairs,
                    did_flip=did_flip,
                )
        image_out = Image.fromarray(augmented["image"])
        if masks_list is not None and "masks" in augmented:
            height, width = augmented["image"].shape[:2]
            masks_aug = augmented["masks"]
            masks_aug = [masks_aug[int(i)] for i in kept_idxs]
            if len(masks_aug) == 0:
                target_out["masks"] = torch.zeros((0, height, width), dtype=torch.bool)
            else:
                target_out["masks"] = torch.as_tensor(np.stack(masks_aug), dtype=torch.bool)
        return image_out, target_out

    def __call__(
        self, image: PIL.Image.Image, target: dict[str, Any] | None
    ) -> tuple[PIL.Image.Image, dict[str, Any] | None]:
        """Apply the Albumentations transform to image and target.

        This method handles the data format conversion between RF-DETR and Albumentations:
        1. Converts PIL Image to numpy array (required by Albumentations)
        2. Converts PyTorch tensors to numpy/lists (required by Albumentations)
        3. Applies the transform
        4. Converts results back to PIL Image and PyTorch tensors

        For geometric transforms with bounding boxes, this method also:
        - Validates box shapes and coordinates
        - Handles boxes that may be removed by the transform (e.g., cropped out)
        - Ensures labels stay synchronized with their corresponding boxes
        - Transforms masks when present to stay aligned with the image

        Args:
            image: Input PIL Image in RGB format.
            target: Target dictionary containing:
                - 'labels': PyTorch tensor of shape (N,) with class labels
                - 'boxes' (optional): PyTorch tensor of shape (N, 4) in (x1, y1, x2, y2) format
                - 'masks' (optional): PyTorch tensor of shape (N, H, W) with instance segmentation masks.
                  For geometric transforms, masks are transformed alongside boxes to maintain alignment. Requires
                  'boxes' to be present; a warning is logged if masks exist without boxes.
                Pass ``None`` for inference scenarios where no ground-truth annotations are available.

        Returns:
            Tuple of (transformed_image, transformed_target):
                - transformed_image: PIL Image after augmentation
                - transformed_target: Dictionary with augmented boxes and labels, or ``None`` if
                  ``target`` was ``None``.

        Raises:
            TypeError: If target is not a dictionary (and not None).
            KeyError: If target doesn't contain 'labels' key.
            ValueError: If boxes don't have shape (N, 4).

        Examples:
            >>> from albumentations import HorizontalFlip
            >>> wrapper = AlbumentationsWrapper(HorizontalFlip(p=1.0))
            >>> image = Image.new('RGB', (100, 100))
            >>> target = {"boxes": torch.tensor([[10, 20, 90, 80]]), "labels": torch.tensor([1])}
            >>> aug_image, aug_target = wrapper(image, target)
        """
        # === Inference mode: no ground-truth annotations ===
        if target is None:
            image_np = np.array(image)
            if self._is_geometric:
                # Geometric A.Compose requires label_fields even when there are no boxes
                augmented = self.transform(
                    image=image_np,
                    bboxes=[],
                    category_ids=[],
                    idxs=[],
                    keypoints=[],
                    keypoint_instance_ids=[],
                    keypoint_point_ids=[],
                    keypoint_visibility=[],
                )
            else:
                augmented = self.transform(image=image_np)
            return Image.fromarray(augmented["image"]), None

        # === Input Validation ===
        if not isinstance(target, dict):
            raise TypeError(f"target must be a dictionary, got {type(target)}")
        if "labels" not in target:
            raise KeyError("target must contain 'labels' key")

        # === Format Conversion: PyTorch/PIL → Albumentations ===
        # Convert PIL Image to numpy array (HWC format expected by Albumentations)
        image_np = np.array(image)

        # Convert labels tensor to Python list (required by Albumentations category_ids)
        labels = target["labels"].cpu().tolist() if torch.is_tensor(target["labels"]) else list(target["labels"])

        # === Apply Transform ===
        if self._is_geometric and "masks" in target and "boxes" not in target:
            logger.warning(
                "AlbumentationsWrapper: geometric transform requested with 'masks' but without 'boxes'. "
                "Masks will not be geometrically transformed because bounding boxes are missing."
            )
        if self._is_geometric and "boxes" in target:
            # Geometric path: transform image and boxes together
            image_out, target_out = self._apply_geometric_transform(image_np, target, labels)
        else:
            # Non-geometric path: transform image only
            augmented = self.transform(image=image_np)
            image_out = Image.fromarray(augmented["image"])
            target_out = target.copy()

        # Ensure 'size' (if present) matches the transformed image size (h, w)
        if "size" in target_out:
            # PIL.Image.size is (width, height); many detectors expect (height, width)
            width, height = image_out.size
            target_out["size"] = torch.as_tensor([height, width], dtype=torch.int64)
        return image_out, target_out

    @staticmethod
    def from_config(
        config_dict: dict[str, Any] | list[dict[str, Any]],
        keypoint_flip_pairs: list[int] | None = None,
        strict: bool = False,
    ) -> list[AlbumentationsWrapper]:
        """Build a list of :class:`AlbumentationsWrapper` instances from a config.

        Supports both a flat dictionary format (backward-compatible) and a list format that allows duplicate transform
        names and explicit ordering. Container transforms (``OneOf``, ``SomeOf``, ``Sequential``) may be nested
        arbitrarily deep.

        **Dict format** (existing, backward-compatible)::

            config = {
                "HorizontalFlip": {"p": 0.5},
                "Rotate": {"limit": 45, "p": 0.3},
                "OneOf": {
                    "transforms": [
                        {"HorizontalFlip": {"p": 1.0}},
                        {"VerticalFlip": {"p": 1.0}},
                    ],
                },
            }

        **List format** (new; useful when you need two entries with the same name or when explicit order matters)::

            config = [
                {"HorizontalFlip": {"p": 0.5}},
                {"OneOf": {
                    "transforms": [
                        {"Rotate": {"limit": 45, "p": 1.0}},
                        {"ShiftScaleRotate": {"p": 1.0}},
                    ],
                }},
            ]

        **Shorthand for container ``transforms`` list** -- when a container key's value is a *list* rather than a dict,
        it is interpreted as the ``transforms`` parameter::

            {"OneOf": [{"HorizontalFlip": {"p": 1.0}}, {"VerticalFlip": {"p": 1.0}}]}

        Args:
            config_dict: Augmentation configuration -- either a ``dict`` mapping
                transform names to parameter dicts, or a ``list`` of single-key dicts ``{name: params}``.
            keypoint_flip_pairs: Joint index pairs for swapping left/right keypoints after a horizontal
                flip (e.g. ``[0, 1, 2, 3]`` swaps joint 0↔1 and 2↔3). Pass ``None`` (default) for
                detection pipelines where horizontal flips are always permitted. Pass an empty list
                ``[]`` to mark a keypoint pipeline without any defined flip pairs -- horizontal-flip
                augmentations are then disabled until flip-pair swapping is implemented.
            strict: When ``True``, a transform that fails to build raises :class:`RuntimeError`
                instead of being logged and skipped. Use for internally-generated pipelines (e.g. the
                required resize stack) where a silently dropped transform would corrupt the output shape.
                Defaults to ``False`` for user augmentation configs, which stay lenient.

        Returns:
            List of :class:`AlbumentationsWrapper` instances in config order.

        Raises:
            ImportError: If Albumentations is not installed.
            TypeError: If *config_dict* is neither a ``dict`` nor a ``list``.
            RuntimeError: If ``strict=True`` and a transform fails to build.

        Examples:
            >>> config = {
            ...     "HorizontalFlip": {"p": 0.5},
            ...     "Rotate": {"limit": 45, "p": 0.3},
            ...     "GaussianBlur": {"p": 0.2}
            ... }
            >>> transforms = AlbumentationsWrapper.from_config(config)
            >>> [t.transform.transforms[0].__class__.__name__ for t in transforms]
            ['HorizontalFlip', 'Rotate', 'GaussianBlur']

        Note:
            Invalid transforms or invalid parameters are logged and skipped gracefully.
        """
        original_config_empty = isinstance(config_dict, (dict, list)) and len(config_dict) == 0
        config_dict = filter_keypoint_hflip_augmentations(
            config_dict,
            include_keypoints=keypoint_flip_pairs is not None and not keypoint_flip_pairs,
            warn=logger.warning,
        )
        if isinstance(config_dict, list):
            entries = config_dict
        elif isinstance(config_dict, dict):
            entries = [{k: v} for k, v in config_dict.items()]
        else:
            raise TypeError(f"config_dict must be a dictionary or list, got {type(config_dict)}")

        if not entries:
            if not original_config_empty:
                return []
            logger.warning("Empty augmentation config provided, no transforms will be applied")
            return []

        if alb is None:
            raise ImportError(
                "Albumentations is required to build RF-DETR dataset transforms. "
                "Install the project dependencies with `uv sync --all-groups` or install albumentations."
            )

        transforms = []
        for entry in entries:
            if not isinstance(entry, dict) or len(entry) != 1:
                logger.warning(
                    "Skipping invalid config entry (must be a single-key dict): %r",
                    entry,
                )
                continue
            aug_name, params = next(iter(entry.items()))

            # Shorthand: container value is a list -> treat as {"transforms": [...]}
            if isinstance(params, list) and aug_name in ALBUMENTATIONS_CONTAINERS:
                params = {"transforms": params}

            if not isinstance(params, dict):
                logger.warning(
                    "Skipping %s: parameters must be a dictionary, got %s",
                    aug_name,
                    type(params).__name__,
                )
                continue

            try:
                transform = _build_albu_transform(aug_name, params)
                transforms.append(AlbumentationsWrapper(transform, keypoint_flip_pairs=keypoint_flip_pairs))
            except Exception as e:
                if strict:
                    raise RuntimeError(f"Failed to build required transform {aug_name!r}: {e}") from e
                logger.warning(
                    "Failed to initialize %s with params %r: %s. Skipping.",
                    aug_name,
                    params,
                    e,
                )
                continue

        logger.info("Built %d Albumentations transforms from config", len(transforms))
        return transforms

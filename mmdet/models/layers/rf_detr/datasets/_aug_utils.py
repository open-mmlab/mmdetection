# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared helpers for dataset augmentation configuration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Transforms that include a horizontal-flip component. Applying these to keypoint
# data without swapping left/right joint pairs produces incorrect annotations.
HFLIP_TRANSFORM_NAMES: frozenset[str] = frozenset({"HorizontalFlip", "Flip", "D4"})

CONTAINER_TRANSFORM_NAMES: frozenset[str] = frozenset({"OneOf", "SomeOf", "Sequential"})


def _warn_keypoint_hflip_disabled(aug_name: str, warn: Callable[..., None]) -> None:
    """Emit the standard warning for a disabled keypoint horizontal flip."""
    warn(
        "Keypoint pipeline: '%s' performs a horizontal flip but no keypoint flip pairs "
        "were configured. The transform has been disabled to prevent incorrect keypoint "
        "annotations. Remove '%s' from your augmentation config or provide keypoint_flip_pairs.",
        aug_name,
        aug_name,
    )


def filter_keypoint_hflip_augmentations(
    config: Any,
    *,
    include_keypoints: bool = True,
    warn: Callable[..., None],
) -> Any:
    """Drop horizontal-flip transforms from keypoint augmentation configs.

    The helper preserves the input config shape: dictionary configs return
    dictionaries, and ordered list configs return lists. Container transforms are
    filtered recursively so nested hflip entries cannot survive inside
    ``OneOf``, ``SomeOf``, or ``Sequential``.

    Args:
        config: Augmentation config accepted by RF-DETR augmentation builders.
        include_keypoints: Whether the config will be applied to keypoint data.
        warn: Warning sink, typically ``logger.warning``.

    Returns:
        A filtered augmentation config with keypoint-unsafe hflip entries removed.
    """
    if not include_keypoints:
        return config

    if isinstance(config, list):
        return _filter_keypoint_hflip_entries(config, warn=warn)

    if isinstance(config, dict):
        return _filter_keypoint_hflip_dict(config, warn=warn)

    return config


def _filter_keypoint_hflip_entries(entries: list[dict[str, Any]], *, warn: Callable[..., None]) -> list[dict[str, Any]]:
    """Filter ordered single-key augmentation entries."""
    filtered_entries: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict) or len(entry) != 1:
            filtered_entries.append(entry)
            continue

        aug_name, params = next(iter(entry.items()))
        filtered_params = _filter_keypoint_hflip_item(aug_name, params, warn=warn)
        if isinstance(filtered_params, _DropAugmentation):
            continue
        filtered_entries.append({aug_name: filtered_params})
    return filtered_entries


def _filter_keypoint_hflip_dict(config: dict[str, Any], *, warn: Callable[..., None]) -> dict[str, Any]:
    """Filter a mapping-style augmentation config."""
    filtered_config: dict[str, Any] = {}
    for aug_name, params in config.items():
        filtered_params = _filter_keypoint_hflip_item(aug_name, params, warn=warn)
        if isinstance(filtered_params, _DropAugmentation):
            continue
        filtered_config[aug_name] = filtered_params
    return filtered_config


class _DropAugmentation:
    """Sentinel used when an augmentation entry should be removed."""


def _filter_keypoint_hflip_item(
    aug_name: str,
    params: Any,
    *,
    warn: Callable[..., None],
) -> Any | _DropAugmentation:
    """Filter one augmentation entry, recursing into supported containers."""
    if aug_name in HFLIP_TRANSFORM_NAMES:
        _warn_keypoint_hflip_disabled(aug_name, warn)
        return _DropAugmentation()

    if aug_name not in CONTAINER_TRANSFORM_NAMES:
        return params

    if isinstance(params, list):
        filtered_transforms = _filter_keypoint_hflip_entries(params, warn=warn)
        return filtered_transforms if filtered_transforms else _DropAugmentation()

    if not isinstance(params, dict) or "transforms" not in params:
        return params

    transforms = params["transforms"]
    if not isinstance(transforms, list):
        return params

    filtered_transforms = _filter_keypoint_hflip_entries(transforms, warn=warn)
    if not filtered_transforms:
        return _DropAugmentation()

    return {**params, "transforms": filtered_transforms}

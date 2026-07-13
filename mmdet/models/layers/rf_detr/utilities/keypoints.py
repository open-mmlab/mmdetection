# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Keypoint utility functions shared by inference and visualization."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "schemas_semantically_equal",
    "precision_cholesky_to_pixel_covariance",
]


def _is_bg_first_schema(schema: list[int]) -> bool:
    """Return True if *schema* uses a background-first layout.

    A background-first schema has a leading slot with zero keypoints that
    acts as a background/no-keypoint sentinel, e.g. ``[0, 17]`` where slot 0
    is background and slot 1 is person with 17 keypoints.  Active-first
    schemas omit that slot entirely, e.g. ``[17]``.

    Args:
        schema: Keypoints-per-class list.

    Returns:
        ``True`` when ``schema`` is non-empty and its first element is zero.

    Examples:
        >>> _is_bg_first_schema([0, 17])
        True
        >>> _is_bg_first_schema([17])
        False
        >>> _is_bg_first_schema([])
        False
    """
    return bool(schema) and schema[0] == 0


def _to_active_first(schema: list[int]) -> list[int]:
    """Strip the leading background slot from a bg-first schema.

    Always returns a new list. A no-op (copy) when *schema* is already
    active-first or empty. Only the first leading zero slot is removed;
    schemas with multiple leading zeros (e.g. ``[0, 0, 17]``) retain all
    but the first.

    Args:
        schema: Keypoints-per-class list.

    Returns:
        Schema with the leading zero-keypoint slot removed when bg-first,
        or a copy of *schema* otherwise.

    Examples:
        >>> _to_active_first([0, 17])
        [17]
        >>> _to_active_first([17])
        [17]
        >>> _to_active_first([0, 17, 4])
        [17, 4]
        >>> _to_active_first([0])
        []
    """
    if _is_bg_first_schema(schema):
        return schema[1:]
    return list(schema)


def _to_bg_first(schema: list[int]) -> list[int]:
    """Prepend a background slot to an active-first schema.

    A no-op when *schema* already starts with a zero-keypoint slot or is empty.

    Args:
        schema: Keypoints-per-class list.

    Returns:
        Schema with a leading ``0`` prepended when not already bg-first.

    Examples:
        >>> _to_bg_first([17])
        [0, 17]
        >>> _to_bg_first([0, 17])
        [0, 17]
        >>> _to_bg_first([17, 4])
        [0, 17, 4]
    """
    if _is_bg_first_schema(schema) or not schema:
        return list(schema)
    return [0] + list(schema)


def schemas_semantically_equal(a: list[int], b: list[int]) -> bool:
    """Return True if *a* and *b* represent the same keypoint structure.

    Two schemas are semantically equal when they encode identical active
    keypoint counts per class after stripping any leading background slot.
    This allows ``[0, 17]`` (bg-first) and ``[17]`` (active-first) to
    compare equal.

    Args:
        a: First keypoints-per-class list.
        b: Second keypoints-per-class list.

    Returns:
        ``True`` when both schemas reduce to the same active-first form.

    Note:
        A schema of ``[0]`` (one detection-only background slot) is semantically
        equal to ``[]`` (no schema) because both reduce to an empty active-first
        form.  Callers that need to distinguish "no schema" from "bg-first with no
        active slots" should compare ``to_active_first(a)`` directly or check
        ``bool(schema)`` before calling this function.

        Use this function for user-facing validation where representational form
        does not matter.  Use exact ``!=`` comparison when preserving
        representational form is required (e.g. auto-align in checkpoint loading).

    Examples:
        >>> schemas_semantically_equal([0, 17], [17])
        True
        >>> schemas_semantically_equal([17], [17])
        True
        >>> schemas_semantically_equal([0, 17], [0, 33])
        False
        >>> schemas_semantically_equal([0], [])
        True
    """
    return _to_active_first(a) == _to_active_first(b)


def precision_cholesky_to_pixel_covariance(
    precision_cholesky: NDArray[np.floating],
    source_shape: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Convert RF-DETR keypoint precision parameters into pixel covariances.

    The keypoint head predicts lower-triangular precision-Cholesky parameters in
    normalized image coordinates. This helper inverts those precision matrices
    and scales them to pixel coordinates for Supervision covariance annotators.

    Args:
        precision_cholesky: Lower-triangular precision parameters with shape
            ``(N, K, 3)``. Each triplet is ``(log_l11, l21, log_l22)``.
        source_shape: Per-detection ``(height, width)`` rows with shape
            ``(N, 2)``.

    Returns:
        Pixel-space covariance matrices with shape ``(N, K, 2, 2)``.

    Raises:
        ValueError: If ``precision_cholesky`` or ``source_shape`` has an
            incompatible shape.

    Example:
        >>> precision = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
        >>> shape = np.array([[10.0, 20.0]], dtype=np.float32)
        >>> precision_cholesky_to_pixel_covariance(precision, shape)[0, 0]
        array([[400.,   0.],
               [  0., 100.]], dtype=float32)
    """
    if precision_cholesky.ndim != 3 or precision_cholesky.shape[2] != 3:
        raise ValueError(f"precision_cholesky must have shape (N, K, 3), got {precision_cholesky.shape}.")
    if source_shape.shape != (precision_cholesky.shape[0], 2):
        raise ValueError(f"source_shape must have shape ({precision_cholesky.shape[0]}, 2), got {source_shape.shape}.")

    # Closed-form 2x2 inverse, fully vectorized — avoids per-keypoint ``np.linalg.inv``
    # dispatch (~50 us each) that dominates runtime for typical N=300, K=17 batches.
    #
    # For lower-triangular Cholesky ``L = [[l11, 0], [l21, l22]]``, the precision is
    # ``P = L @ L.T`` with closed-form inverse:
    #
    #     det(P) = (l11 * l22) ** 2
    #     cov[0, 0] = (l21 ** 2 + l22 ** 2) / det
    #     cov[0, 1] = -l11 * l21 / det
    #     cov[1, 1] = l11 ** 2 / det
    #
    # Pixel-space scaling ``diag([width, height]) @ cov @ diag([width, height])`` then
    # gives ``px[0, 0] = width ** 2 * cov[0, 0]``, ``px[1, 1] = height ** 2 * cov[1, 1]``,
    # ``px[0, 1] = width * height * cov[0, 1]``.
    #
    # Non-finite Cholesky inputs and degenerate precision (det -> 0, producing inf/nan
    # pixel covariances) are masked to NaN — replaces the original ``LinAlgError`` /
    # ``isfinite`` guards from the per-keypoint loop.
    precision_cholesky_f64 = precision_cholesky.astype(np.float64, copy=False)
    log_l11 = precision_cholesky_f64[..., 0]
    l21 = precision_cholesky_f64[..., 1]
    log_l22 = precision_cholesky_f64[..., 2]

    finite_input = np.isfinite(precision_cholesky_f64).all(axis=-1)  # (N, K)

    l11 = np.exp(log_l11)
    l22 = np.exp(log_l22)

    # ``det(precision) = det(L) ** 2 = (l11 * l22) ** 2``; using ``errstate`` to suppress
    # the expected ``divide``/``invalid`` warnings for degenerate keypoints — finite mask
    # below replaces the resulting inf/nan entries with NaN.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inv_det = 1.0 / (l11 * l11 * l22 * l22)
        cov00 = inv_det * (l21 * l21 + l22 * l22)
        # Add 0.0 to normalise IEEE-754 ``-0.0`` to ``+0.0`` when l21 vanishes — otherwise
        # the off-diagonal would render as ``-0.`` in the doctest output and in user-facing
        # arrays without changing any non-degenerate result.
        cov01 = inv_det * (-l11 * l21) + 0.0
        cov11 = inv_det * (l11 * l11)

        # Broadcast source_shape (N, 2) over keypoint axis via (N, 1) reshape.
        height = source_shape[:, 0:1].astype(np.float64)
        width = source_shape[:, 1:2].astype(np.float64)

        px00 = width * width * cov00
        px01 = width * height * cov01
        px11 = height * height * cov11

    finite_all = finite_input & np.isfinite(px00) & np.isfinite(px01) & np.isfinite(px11)
    nan = np.float32(np.nan)

    covariances = np.full((*precision_cholesky.shape[:2], 2, 2), nan, dtype=np.float32)
    covariances[..., 0, 0] = np.where(finite_all, px00.astype(np.float32), nan)
    covariances[..., 0, 1] = np.where(finite_all, px01.astype(np.float32), nan)
    covariances[..., 1, 0] = covariances[..., 0, 1]
    covariances[..., 1, 1] = np.where(finite_all, px11.astype(np.float32), nan)
    return covariances

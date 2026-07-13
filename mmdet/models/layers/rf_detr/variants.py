# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Concrete RF-DETR model variant classes.

All classes inherit from :class:`~rfdetr.detr.RFDETR` which remains defined in ``rfdetr.detr``. Backward-compatible
access from ``rfdetr.detr`` is provided via lazy ``__getattr__`` re-exports, so importing ``rfdetr.variants`` no longer
depends on a fragile eager ``detr -> variants`` import sequence.
"""

from __future__ import annotations

__all__ = [
    "RFDETRBase",
    "RFDETRKeypointPreview",
    "RFDETRNano",
    "RFDETRSmall",
    "RFDETRMedium",
    "RFDETRLarge",
    "RFDETRLargeDeprecated",
    "RFDETRSeg",
    "RFDETRSegPreview",
    "RFDETRSegNano",
    "RFDETRSegSmall",
    "RFDETRSegMedium",
    "RFDETRSegLarge",
    "RFDETRSegXLarge",
    "RFDETRSeg2XLarge",
]

from deprecate import deprecated_class

from mmdet.models.layers.rf_detr.config import (
    KeypointTrainConfig,
    ModelConfig,
    RFDETRBaseConfig,
    RFDETRKeypointPreviewConfig,
    RFDETRLargeConfig,
    RFDETRLargeDeprecatedConfig,
    RFDETRMediumConfig,
    RFDETRNanoConfig,
    RFDETRSeg2XLargeConfig,
    RFDETRSegLargeConfig,
    RFDETRSegMediumConfig,
    RFDETRSegNanoConfig,
    RFDETRSegPreviewConfig,
    RFDETRSegSmallConfig,
    RFDETRSegXLargeConfig,
    RFDETRSmallConfig,
    SegmentationTrainConfig,
)
from mmdet.models.layers.rf_detr.detr import RFDETR
from mmdet.models.layers.rf_detr.utilities.logger import get_logger

logger = get_logger()


@deprecated_class(
    target=None,
    deprecated_in="1.7.0",
    remove_in="2.0.0",
)
class RFDETRBase(RFDETR):
    """Train an RF-DETR Base model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``.
    """

    size = "rfdetr-base"
    _model_config_class = RFDETRBaseConfig


class RFDETRNano(RFDETR):
    """Train an RF-DETR Nano model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``.
    """

    size = "rfdetr-nano"
    _model_config_class = RFDETRNanoConfig


class RFDETRKeypointPreview(RFDETR):
    """Train or run inference with the RF-DETR keypoint preview model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``.
    """

    size = "rfdetr-keypoint-preview"
    _model_config_class = RFDETRKeypointPreviewConfig
    _train_config_class = KeypointTrainConfig


class RFDETRSmall(RFDETR):
    """Train an RF-DETR Small model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``.
    """

    size = "rfdetr-small"
    _model_config_class = RFDETRSmallConfig


class RFDETRMedium(RFDETR):
    """Train an RF-DETR Medium model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``.
    """

    size = "rfdetr-medium"
    _model_config_class = RFDETRMediumConfig


@deprecated_class(
    target=None,
    deprecated_in="1.7.0",
    remove_in="2.0.0",
)
class RFDETRLargeDeprecated(RFDETR):
    """Train an RF-DETR Large model using the legacy config.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``.
    """

    size = "rfdetr-large"
    _model_config_class = RFDETRLargeDeprecatedConfig


class RFDETRLarge(RFDETR):
    """Train an RF-DETR Large model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``.
    """

    size = "rfdetr-large"

    @staticmethod
    def _should_fallback_to_deprecated_config(exc: Exception) -> bool:
        """Return whether initialization should retry with deprecated Large config.

        The fallback is only for known checkpoint/config incompatibilities from deprecated Large weights. Runtime issues
        such as CUDA OOM must fail fast and must not trigger a second initialization attempt.

        Args:
            exc: Exception raised by initial ``RFDETR`` initialization.

        Returns:
            ``True`` when retrying with deprecated config is expected to help.
        """
        message = str(exc).lower()
        if "out of memory" in message:
            return False
        if isinstance(exc, ValueError):
            return "patch_size" in message
        if isinstance(exc, RuntimeError):
            incompatible_state_dict_markers = (
                "error(s) in loading state_dict",
                "size mismatch",
                "missing key(s) in state_dict",
                "unexpected key(s) in state_dict",
            )
            return any(marker in message for marker in incompatible_state_dict_markers)
        return False

    def __init__(self, **kwargs):
        self.init_error = None
        self.is_deprecated = False
        # When the user explicitly sets a custom resolution, a PE size mismatch
        # is caused by the resolution change — not by deprecated weights.  Guard
        # against the fallback heuristic misclassifying it as deprecated weights.
        # Only suppress the fallback when the provided resolution genuinely differs
        # from the class default; passing resolution=<default> explicitly (e.g. from
        # a serialised config round-trip) must still allow the deprecated-weights retry.
        _default_resolution = RFDETRLargeConfig.model_fields["resolution"].default
        _custom_resolution = "resolution" in kwargs and kwargs.get("resolution") != _default_resolution
        try:
            super().__init__(**kwargs)
        except (ValueError, RuntimeError) as exc:
            if _custom_resolution or not self._should_fallback_to_deprecated_config(exc):
                raise
            self.init_error = exc
            self.is_deprecated = True
            try:
                super().__init__(**kwargs)
                logger.warning(
                    "\n"
                    "=" * 100 + "\n"
                    "WARNING: Automatically switched to deprecated model configuration,"
                    " due to using deprecated weights."
                    " This will be removed in v1.9.0.\n"
                    " Please retrain your model with the new weights and configuration.\n"
                    "=" * 100 + "\n"
                )
            except Exception as retry_exc:
                logger.exception(
                    "Retry with deprecated RF-DETR Large configuration failed; "
                    "re-raising the original initialization error for compatibility. "
                    "Original error: %s",
                    self.init_error,
                    exc_info=retry_exc,
                )
                raise self.init_error from retry_exc

    def get_model_config(self, **kwargs) -> ModelConfig:
        if not self.is_deprecated:
            return RFDETRLargeConfig(**kwargs)
        else:
            return RFDETRLargeDeprecatedConfig(**kwargs)


class RFDETRSeg(RFDETR):
    """Base class for all RF-DETR segmentation models.

    Training accepts custom square integer ``resolution`` values. Most segmentation variants use multiples of 24;
    ``RFDETRSegNano`` uses multiples of 12.
    """

    _train_config_class = SegmentationTrainConfig


@deprecated_class(
    target=None,
    deprecated_in="1.7.0",
    remove_in="2.0.0",
)
class RFDETRSegPreview(RFDETRSeg):
    """Train an RF-DETR Segmentation Preview model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``. Deprecated in v1.7.0, scheduled for removal in v2.0.0.
    """

    size = "rfdetr-seg-preview"
    _model_config_class = RFDETRSegPreviewConfig


class RFDETRSegNano(RFDETRSeg):
    """Train an RF-DETR Segmentation Nano model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``; this variant uses multiples of 12.
    """

    size = "rfdetr-seg-nano"
    _model_config_class = RFDETRSegNanoConfig


class RFDETRSegSmall(RFDETRSeg):
    """Train an RF-DETR Segmentation Small model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``; this variant uses multiples of 24.
    """

    size = "rfdetr-seg-small"
    _model_config_class = RFDETRSegSmallConfig


class RFDETRSegMedium(RFDETRSeg):
    """Train an RF-DETR Segmentation Medium model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``; this variant uses multiples of 24.
    """

    size = "rfdetr-seg-medium"
    _model_config_class = RFDETRSegMediumConfig


class RFDETRSegLarge(RFDETRSeg):
    """Train an RF-DETR Segmentation Large model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``; this variant uses multiples of 24.
    """

    size = "rfdetr-seg-large"
    _model_config_class = RFDETRSegLargeConfig


class RFDETRSegXLarge(RFDETRSeg):
    """Train an RF-DETR Segmentation XLarge model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``; this variant uses multiples of 24.
    """

    size = "rfdetr-seg-xlarge"
    _model_config_class = RFDETRSegXLargeConfig


class RFDETRSeg2XLarge(RFDETRSeg):
    """Train an RF-DETR Segmentation 2XLarge model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``; this variant uses multiples of 24.
    """

    size = "rfdetr-seg-2xlarge"
    _model_config_class = RFDETRSeg2XLargeConfig

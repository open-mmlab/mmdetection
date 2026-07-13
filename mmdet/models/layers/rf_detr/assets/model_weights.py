# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Model weights abstraction and download system.

Provides forward-compatible pattern for model weights across rf-detr and rf-detr-plus packages. External packages (like
rf-detr-plus) should inherit from ModelWeightsBase to ensure compile-time interface compatibility.

Critical Strategic Decisions:
    1. **Standalone first**: Check local ModelWeights before lazy-importing external packages
    2. **Compile-time safety**: Inheritance-based compatibility via ModelWeightsBase
    3. **Clean abstraction**: Enum values ARE ModelWeightAsset dataclass instances
    4. **Backward compatible**: Legacy OPEN_SOURCE_MODELS dict maintained
    5. **Offline testable**: All I/O operations mockable

Download Priority Order:
    1. Local ModelWeights.from_filename() - rf-detr's built-in models
    2. rfdetr_plus.assets.ModelWeights.from_filename() - lazy import if not found locally
    3. PLATFORM_MODELS dict - legacy fallback for backward compatibility
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from mmdet.models.layers.rf_detr.platform import _IS_RFDETR_PLUS_AVAILABLE
from mmdet.models.layers.rf_detr.utilities.files import _download_file, _validate_file_md5
from mmdet.models.layers.rf_detr.utilities.logger import get_logger

logger = get_logger()

_RF_HOME_ENV_VAR = "RF_HOME"
_DEFAULT_CACHE_DIR = "~/.roboflow/models"


@dataclass(frozen=True)
class ModelWeightAsset:
    """Dataclass representing a model asset with download information.

    This is the standard format for model assets across rf-detr packages. Both rf-detr and rf-detr-plus should use this
    structure for compatibility.

    Attributes:
        filename: The local filename for the model weights
        url: The download URL
        md5_hash: The expected MD5 hash for integrity validation (None if not available)

    Example:
        >>> asset = ModelWeightAsset(
        ...     filename='rf-detr-base.pth',
        ...     url='https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth',
        ...     md5_hash='b4d3ce46099eaed50626ede388caf979'
        ... )
    """

    filename: str
    url: str
    md5_hash: str | None = None


class ModelWeightsBase(Enum):
    """Base class for model weight registries.

    This base class ensures compile-time compatibility between rf-detr and rf-detr-plus. Both packages should inherit
    from this class to ensure they have the same interface.

    Each enum member's value must be a ModelWeightAsset instance.

    Example inheritance:
        >>> from mmdet.models.layers.rf_detr.assets import ModelWeightAsset
        >>> class MyModelWeights(ModelWeightsBase):
        ...     MODEL_NAME = ModelWeightAsset(
        ...         "model.pth",
        ...         "https://example.com/model.pth",
        ...         "abc123"
        ...     )

    Example usage:
        >>> from mmdet.models.layers.rf_detr.assets.model_weights import ModelWeights
        >>> asset = ModelWeights.from_filename("rf-detr-base.pth")
        >>> asset.filename
        'rf-detr-base.pth'
        >>> asset.url  # doctest: +SKIP
        'https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth'
    """

    def __new__(cls, asset: ModelWeightAsset) -> ModelWeightsBase:
        obj = object.__new__(cls)
        obj._value_ = asset
        return obj

    # Convenience properties to access the underlying ModelWeightAsset attributes
    @property
    def filename(self) -> str:
        """Get the filename from the underlying ModelWeightAsset."""
        return self.value.filename

    @property
    def url(self) -> str:
        """Get the URL from the underlying ModelWeightAsset."""
        return self.value.url

    @property
    def md5_hash(self) -> str | None:
        """Get the MD5 hash from the underlying ModelWeightAsset."""
        return self.value.md5_hash

    @classmethod
    def from_filename(cls, filename: str) -> ModelWeightAsset | None:
        """Get ModelWeightAsset by filename.

        Args:
            filename: The model filename (e.g., 'rf-detr-base.pth')

        Returns:
            ModelWeightAsset instance if found, None otherwise

        Example:
            >>> asset = ModelWeights.from_filename('rf-detr-base.pth')
            >>> asset.url
            'https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth'
        """
        for member in cls:
            if member.value.filename == filename:
                return member.value  # Return the ModelWeightAsset directly
        return None

    @classmethod
    def get_url(cls, filename: str) -> str | None:
        """Get download URL for a model by filename.

        Args:
            filename: The model filename

        Returns:
            URL string if found, None otherwise
        """
        asset = cls.from_filename(filename)
        return asset.url if asset else None

    @classmethod
    def get_md5(cls, filename: str) -> str | None:
        """Get expected MD5 hash for a model by filename.

        Args:
            filename: The model filename

        Returns:
            MD5 hash string if available, None otherwise
        """
        asset = cls.from_filename(filename)
        return asset.md5_hash if asset else None

    @classmethod
    def list_models(cls) -> list[str]:
        """List all available model filenames.

        Returns:
            List of model filenames
        """
        return [member.value.filename for member in cls]


class ModelWeights(ModelWeightsBase):
    """Enumeration of available RF-DETR model assets.

    Inherits from ModelWeightsBase to ensure compatibility with rf-detr-plus.

    Each enum member's value is a ModelWeightAsset instance containing:
    - filename: The local filename for the model weights
    - url: The download URL
    - md5_hash: The expected MD5 hash for integrity validation

    Example:
        >>> asset = ModelWeights.RF_DETR_BASE
        >>> asset.filename
        'rf-detr-base.pth'
        >>> asset.url
        'https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth'
    """

    # Detection Models
    RF_DETR_BASE = ModelWeightAsset(
        "rf-detr-base.pth",
        "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
        "b4d3ce46099eaed50626ede388caf979",
    )
    RF_DETR_BASE_O365 = ModelWeightAsset(
        "rf-detr-base-o365.pth",
        "https://storage.googleapis.com/rfdetr/top-secret-1234/lwdetr_dinov2_small_o365_checkpoint.pth",
        "d93f4921ccbb0f0a2e4364bed290892b",
    )
    RF_DETR_BASE_2 = ModelWeightAsset(
        "rf-detr-base-2.pth",
        "https://storage.googleapis.com/rfdetr/rf-detr-base-2.pth",
        "462f4d9df407ddc1812f42614040e913",
    )
    RF_DETR_LARGE = ModelWeightAsset(
        "rf-detr-large.pth",
        "https://storage.googleapis.com/rfdetr/rf-detr-large.pth",
        "992c8e862aa733a7bb2777e45d49f1a0",
    )
    RF_DETR_LARGE_2026 = ModelWeightAsset(
        "rf-detr-large-2026.pth",
        "https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth",
        "5cb72153541cbcb9aa6efa26222acc75",
    )
    RF_DETR_NANO = ModelWeightAsset(
        "rf-detr-nano.pth",
        "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
        "fb6504cce7fbdc783f7a46991f07639f",
    )
    RF_DETR_SMALL = ModelWeightAsset(
        "rf-detr-small.pth",
        "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
        "fb37061c1af7bace359c91b723a8d5c1",
    )
    RF_DETR_MEDIUM = ModelWeightAsset(
        "rf-detr-medium.pth",
        "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",
        "7223f764a87b863f02eb8d52bf0ce2ee",
    )
    RF_DETR_KEYPOINT_PREVIEW = ModelWeightAsset(
        "rf-detr-keypoint-preview-xlarge.pth",
        "https://storage.googleapis.com/rfdetr/rf-detr-keypoint-preview-xlarge.pth",
        "6de511943ee85a547d4c5cb527daf0eb",
    )

    # Segmentation Models
    RF_DETR_SEG_PREVIEW = ModelWeightAsset(
        "rf-detr-seg-preview.pt",
        "https://storage.googleapis.com/rfdetr/rf-detr-seg-preview.pt",
        "e35820c28fb86080558123e47a5e49ca",
    )
    RF_DETR_SEG_NANO = ModelWeightAsset(
        "rf-detr-seg-nano.pt",
        "https://storage.googleapis.com/rfdetr/rf-detr-seg-n-ft.pth",
        "9995497791d0ff1664a1d9ddee9cfd20",
    )
    RF_DETR_SEG_SMALL = ModelWeightAsset(
        "rf-detr-seg-small.pt",
        "https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth",
        "0a2a3006381d0c42853907e700eadd08",
    )
    RF_DETR_SEG_MEDIUM = ModelWeightAsset(
        "rf-detr-seg-medium.pt",
        "https://storage.googleapis.com/rfdetr/rf-detr-seg-m-ft.pth",
        "a49af1562c3719227ad43d0ca53b4c7a",
    )
    RF_DETR_SEG_LARGE = ModelWeightAsset(
        "rf-detr-seg-large.pt",
        "https://storage.googleapis.com/rfdetr/rf-detr-seg-l-ft.pth",
        "275f7b094909544ed2841c94a677d07e",
    )
    RF_DETR_SEG_XLARGE = ModelWeightAsset(
        "rf-detr-seg-xlarge.pt",
        "https://storage.googleapis.com/rfdetr/rf-detr-seg-xl-ft.pth",
        "3693b35d0eea86ebb3e0444f4a611fba",
    )
    RF_DETR_SEG_XXLARGE = ModelWeightAsset(
        "rf-detr-seg-xxlarge.pt",
        "https://storage.googleapis.com/rfdetr/rf-detr-seg-2xl-ft.pth",
        "040bc3412af840fa8a47e0ff69b552ba",
    )
    # All methods inherited from ModelWeightsBase


def get_model_cache_dir() -> str:
    """Return the directory where RF-DETR caches downloaded model weights.

    Reads the ``RF_HOME`` environment variable; defaults to ``~/.roboflow/models`` when the variable is not set.

    Set ``RF_HOME`` to override the cache location for all RF-DETR models:

    .. code-block:: bash

        export RF_HOME=/mnt/shared/models

    Args: None

    Returns:
        Absolute, user-expanded path to the model cache directory.  The directory is *not* created by this function —
        callers that need it to exist should call ``os.makedirs(get_model_cache_dir(), exist_ok=True)`` themselves.

    Examples:
        >>> import os
        >>> _ = os.environ.pop("RF_HOME", None)  # ensure default
        >>> expected = os.path.normpath(os.path.expanduser("~/.roboflow/models"))
        >>> get_model_cache_dir() == expected
        True
        >>> os.environ["RF_HOME"] = "~/rfdetr_cache"
        >>> expected = os.path.normpath(os.path.expanduser("~/rfdetr_cache"))
        >>> get_model_cache_dir() == expected
        True
        >>> del os.environ["RF_HOME"]
    """
    cache_dir = os.environ.get(_RF_HOME_ENV_VAR, _DEFAULT_CACHE_DIR)
    return os.path.abspath(os.path.expanduser(cache_dir))


def download_pretrain_weights(
    pretrain_weights: str,
    redownload: bool = False,
    validate_md5: bool = True,
) -> None:
    """Download pretrained weights with optional MD5 validation.

    Download Priority Order:
        The function searches for models in the following order, stopping at the first match:

        1. **Local ModelWeights** (primary source):
           - Checks rf-detr's built-in ModelWeights enum
           - Ensures rf-detr works completely standalone
           - No unnecessary imports or performance overhead

        2. **External packages** (lazy import):
           - Only attempts import if model not found locally
           - Tries rf-detr-plus.assets.ModelWeights if installed
           - Gracefully handles missing packages (ImportError/AttributeError)

        3. **Legacy platform models** (backward compatibility):
           - Falls back to PLATFORM_MODELS dict for older models
           - Maintains compatibility with existing deployments
           - No MD5 validation for legacy models

    Args:
        pretrain_weights: Name of the pretrained weights file (e.g., 'rf-detr-base.pth')
        redownload: Force re-download even if file exists
        validate_md5: Whether to validate MD5 hash of downloaded file

    Example:
        >>> download_pretrain_weights('rf-detr-base.pth')  # doctest: +SKIP
        Downloading pretrained weights for rf-detr-base.pth
    """
    asset: ModelWeightAsset | None = None

    # Use basename for registry lookup so absolute paths (e.g.
    # "/content/rf-detr-base.pth") match the registered short name.
    model_name = os.path.basename(pretrain_weights)

    # First, check local ModelWeights - rf-detr works standalone
    asset = ModelWeights.from_filename(model_name)

    # Only try rf-detr-plus if not found locally (lazy import)
    if asset is None and _IS_RFDETR_PLUS_AVAILABLE:
        try:
            from rfdetr_plus.assets import ModelWeights as PlusModelWeights

            asset = PlusModelWeights.from_filename(model_name)
        except ModuleNotFoundError as ex:
            if ex.name not in {"rfdetr_plus", "rfdetr_plus.assets"}:
                raise

    # Extract URL and MD5 from the asset if found
    if asset is not None:
        url = asset.url
        expected_md5 = asset.md5_hash if validate_md5 else None
    else:
        # If still not found, fall back to legacy dict-based platform models
        try:
            from mmdet.models.layers.rf_detr.platform.downloads import PLATFORM_MODELS

            if model_name not in PLATFORM_MODELS:
                return

            url = PLATFORM_MODELS[model_name]
            expected_md5 = None  # Platform models don't have MD5 hashes yet
        except (ImportError, KeyError):
            return

    # Skip download when file already exists and redownload is disabled
    if os.path.exists(pretrain_weights) and not redownload:
        if expected_md5 and validate_md5:
            if not _validate_file_md5(pretrain_weights, expected_md5):
                logger.warning(
                    f"Existing file {pretrain_weights} has incorrect MD5 hash. "
                    "It may be a user-provided checkpoint or a corrupted/tampered file — "
                    "skipping re-download to avoid overwriting it. "
                    "To force a fresh download of the original weights, pass redownload=True."
                )
            else:
                logger.info(f"File {pretrain_weights} already exists with correct MD5 hash.")
        return

    logger.info(f"Downloading pretrained weights for {pretrain_weights}")
    _download_file(
        url=url,
        filename=pretrain_weights,
        expected_md5=expected_md5,
    )


def validate_pretrain_weights(pretrain_weights: str, strict: bool = False) -> bool:
    """Validate MD5 hash of pretrained weights file.

    Args:
        pretrain_weights: Path to the pretrained weights file
        strict: If True, raise error on validation failure. If False, just warn.

    Returns:
        True if validation passes or no hash is available, False otherwise

    Raises:
        ValueError: If strict=True and validation fails
        FileNotFoundError: If strict=True and file doesn't exist
    """
    if not os.path.exists(pretrain_weights):
        if strict:
            raise FileNotFoundError(f"Pretrained weights file not found: {pretrain_weights}")
        return False

    # Check if we have a hash for this model
    model_name = os.path.basename(pretrain_weights)
    asset = ModelWeights.from_filename(model_name)

    if asset is None or asset.md5_hash is None:
        # No hash available for validation
        logger.debug(f"No MD5 hash available for {model_name}, skipping validation")
        return True

    if not _validate_file_md5(pretrain_weights, asset.md5_hash):
        error_msg = (
            f"MD5 hash validation failed for {pretrain_weights}. "
            f"The file may be corrupted or tampered with. "
            f"Consider re-downloading with download_pretrain_weights('{model_name}', redownload=True)"
        )
        if strict:
            raise ValueError(error_msg)
        else:
            logger.warning(error_msg)
        return False

    logger.debug(f"MD5 validation passed for {pretrain_weights}")
    return True

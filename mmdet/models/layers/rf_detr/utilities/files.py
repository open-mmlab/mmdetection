# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""File download and MD5 validation helpers."""

import hashlib
import os
import tempfile

import requests
from tqdm.auto import tqdm

from mmdet.models.layers.rf_detr.utilities.logger import get_logger

logger = get_logger()
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 30.0


def _compute_file_md5(filepath: str) -> str:
    """Compute MD5 hash of a file.

    Args:
        filepath: Path to the file.

    Returns:
        MD5 hash as hexadecimal string.
    """
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def _validate_file_md5(filepath: str, expected_md5: str) -> bool:
    """Validate that a file's MD5 hash matches the expected hash.

    Args:
        filepath: Path to the file.
        expected_md5: Expected MD5 hash.

    Returns:
        True if hash matches, False otherwise.
    """
    if not os.path.exists(filepath):
        return False

    actual_md5 = _compute_file_md5(filepath)
    return actual_md5.lower() == expected_md5.lower()


def _download_file(
    url: str,
    filename: str,
    expected_md5: str | None = None,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
) -> None:
    """Download a file from a URL with optional MD5 validation.

    Args:
        url: URL to download from.
        filename: Path to save the file.
        expected_md5: Expected MD5 hash for validation (optional).
        timeout: Timeout in seconds passed to ``requests.get``.

    Raises:
        ValueError: If MD5 validation fails.
    """
    # Check if file exists and has correct hash
    if os.path.exists(filename) and expected_md5:
        if _validate_file_md5(filename, expected_md5):
            logger.info(f"File {filename} already exists with correct MD5 hash. Skipping download.")
            return
        else:
            logger.warning(f"File {filename} exists but MD5 hash mismatch. Re-downloading...")
            os.remove(filename)

    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    total_size_header = response.headers.get("content-length")
    try:
        total_size = int(total_size_header) if total_size_header is not None else None
    except (TypeError, ValueError):
        total_size = None

    # Download to a unique temporary file in the destination directory first.
    # This avoids collisions when multiple workers download the same weight file.
    target_dir = os.path.dirname(filename) or "."
    fd, temp_filename = tempfile.mkstemp(
        prefix=f"{os.path.basename(filename)}.",
        suffix=".tmp",
        dir=target_dir,
    )
    try:
        with (
            os.fdopen(fd, "wb") as f,
            tqdm(
                desc=filename,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
    except Exception:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise

    # Validate MD5 if expected hash is provided.
    if expected_md5:
        actual_md5 = _compute_file_md5(temp_filename)
        if actual_md5.lower() != expected_md5.lower():
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            raise ValueError("MD5 mismatch for %s (expected %s, got %s)." % (filename, expected_md5, actual_md5))
        else:
            logger.info(f"MD5 validation successful for {filename}")

    # Atomic replace in target directory.
    try:
        os.replace(temp_filename, filename)
    except Exception:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise

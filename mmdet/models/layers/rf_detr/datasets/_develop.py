# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Private developer tools for testing and benchmarking RF-DETR.

These utilities are intended for internal use by developers and test suites. They are not part of the public API and may
change without notice.
"""

from __future__ import annotations

import os
import shutil
import time
import zipfile
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.request import urlretrieve

import numpy as np
import torch
from PIL import Image

from mmdet.models.layers.rf_detr.util.logger import get_logger

logger = get_logger()

if TYPE_CHECKING:
    import torch

_COCO_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}
_COCO_VAL_IMAGE_COUNT: int = 5000


def _coco_val_images_complete(images_dir: Path) -> bool:
    """Check whether the COCO val2017 image directory contains the expected number of JPEG files.

    Returns ``False`` for a missing or empty directory so callers can trigger a
    re-download without inspecting the directory manually.

    Args:
        images_dir: Path to the directory that should contain the val2017 images.

    Returns:
        ``True`` if *images_dir* exists and contains at least ``_COCO_VAL_IMAGE_COUNT``
        ``.jpg`` files, ``False`` otherwise.

    Raises:
        OSError: If *images_dir* exists but cannot be read (e.g. ``PermissionError``
            when the directory is not accessible, or ``FileNotFoundError`` on a
            TOCTOU race between the ``is_dir()`` check and ``iterdir()``).

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     _coco_val_images_complete(Path(tmpdir) / "val2017")
        False
    """
    if not images_dir.is_dir():
        return False
    return (
        sum(1 for entry in images_dir.iterdir() if entry.is_file() and entry.suffix.lower() == ".jpg")
        >= _COCO_VAL_IMAGE_COUNT
    )


def _nonempty_file_exists(path: Path) -> bool:
    """Check whether a file exists and contains at least one byte.

    Returns ``False`` for a missing or empty file so callers can trigger a
    re-download without inspecting the file manually.

    Args:
        path: Path to the file to check.

    Returns:
        ``True`` if *path* refers to an existing file with ``size > 0``,
        ``False`` otherwise.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     _nonempty_file_exists(Path(tmpdir) / "missing.json")
        False
    """
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def get_coco_download_url(asset: Literal["train2017", "val2017", "annotations"]) -> str:
    """Return the official COCO 2017 download URL for the requested asset."""
    return _COCO_URLS[asset]


class _SimpleDataset:
    """Simple synthetic dataset for testing augmentations and training loops.

    Creates synthetic images with varying numbers of bounding boxes to test edge cases in augmentation pipelines,
    particularly the case where num_boxes=2 (which matches orig_size shape [2]).

    Implements the ``__len__`` / ``__getitem__`` protocol expected by ``torch.utils.data.DataLoader`` without inheriting
    from ``torch.utils.data.Dataset``, so importing this class does not pull in torch at module load time.

    Args:
        num_samples: Number of samples in the dataset.
        transforms: Optional transforms to apply (e.g., Compose of AlbumentationsWrapper).

    Examples:
        >>> from albumentations import HorizontalFlip
        >>> from torchvision.transforms.v2 import Compose
        >>> from mmdet.models.layers.rf_detr.datasets.transforms import AlbumentationsWrapper
        >>>
        >>> transforms = Compose([
        ...     AlbumentationsWrapper(HorizontalFlip(p=0.5)),
        ... ])
        >>> dataset = _SimpleDataset(num_samples=10, transforms=transforms)
        >>> image, target = dataset[0]
    """

    def __init__(self, num_samples: int = 10, transforms: Any | None = None) -> None:
        self.num_samples = num_samples
        self.transforms = transforms

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        # Create synthetic image
        image = Image.new("RGB", (640, 480))

        # Create synthetic target with varying number of boxes
        # Cycles through 1, 2, and 3 boxes to test different edge cases
        num_boxes = (idx % 3) + 1

        boxes = []
        labels = []
        for i in range(num_boxes):
            x1 = 10 + i * 100
            y1 = 10 + i * 50
            x2 = x1 + 80
            y2 = y1 + 100
            boxes.append([x1, y1, x2, y2])
            labels.append(i + 1)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "orig_size": torch.tensor([480, 640]),
            "size": torch.tensor([480, 640]),
            "image_id": torch.tensor([idx]),
            "area": torch.tensor([100.0] * num_boxes),
            "iscrowd": torch.tensor([0] * num_boxes),
        }

        # Apply transforms if any
        if self.transforms:
            image, target = self.transforms(image, target)

        # Convert PIL Image to tensor
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        return image, target


def _download_and_extract(url: str, dest_dir: Path) -> None:
    """Download a zip file and safely extract it into the destination directory.

    Args:
        url: URL to a zip archive.
        dest_dir: Directory where the archive will be saved and extracted.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / url.rsplit("/", 1)[-1]
    logger.info("Downloading %s ...", url)
    urlretrieve(url, str(zip_path))
    logger.info("Extracting %s ...", zip_path)
    dest_dir_resolved = dest_dir.resolve()
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        for member in zf.infolist():
            if not member.filename:
                continue
            target_path = (dest_dir_resolved / member.filename).resolve()
            if not target_path.is_relative_to(dest_dir_resolved):
                raise RuntimeError(f"Unsafe path detected in ZIP file: {member.filename!r}")
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, "r") as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    with suppress(FileNotFoundError):
        zip_path.unlink()


@contextmanager
def _download_lock(lock_path: Path, timeout_s: float = 600.0, poll_s: float = 0.5) -> Generator[None, Any, None]:
    """Provide a simple cross-process lock using an exclusive lock file.

    Args:
        lock_path: Path to the lock file used for mutual exclusion.
        timeout_s: Maximum time in seconds to wait for the lock.
        poll_s: Sleep interval in seconds between lock attempts.

    Yields:
        None. The caller runs inside the locked region.

    Raises:
        TimeoutError: If the lock cannot be acquired within the timeout.

    Note:
        Lock identity is file existence (``O_CREAT | O_EXCL``), not a held file descriptor.
        A SIGKILL-terminated process will leak the lock file until ``timeout_s`` expires.
        If all worker processes have exited but the lock persists, remove it manually:
        ``rm <lock_path>``.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    while True:
        try:
            # Atomic create; raises FileExistsError if another worker owns the lock.
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
            time.sleep(poll_s)
    try:
        yield
    finally:
        # Best-effort cleanup if the lock file was already removed.
        with suppress(FileNotFoundError):
            os.unlink(lock_path)

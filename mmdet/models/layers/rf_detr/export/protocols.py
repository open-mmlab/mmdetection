# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Exporter protocol — enables dependency-inversion for model export."""

from pathlib import Path
from typing import Protocol, runtime_checkable

from torch import nn


@runtime_checkable
class ExporterProtocol(Protocol):
    """Protocol for model exporters.

    Any callable or class that matches this signature can be registered as an exporter without inheriting from a base
    class.
    """

    def __call__(
        self,
        model: nn.Module,
        input_shape: tuple[int, ...],
        output_path: Path,
    ) -> Path:
        """Export *model* to *output_path*.

        Args:
            model: The PyTorch model to export.
            input_shape: Input tensor shape (excluding batch dimension).
            output_path: Destination path for the exported artifact.

        Returns:
            The resolved path to the exported file.
        """
        ...

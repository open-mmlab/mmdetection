# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Deprecation utilities and decorators."""

import warnings

from deprecate import TargetMode, deprecated, void

__all__ = ["TargetMode", "deprecated", "void"]


def _warn_deprecated_module(old: str, new: str, deprecated_in: str, remove_in: str) -> None:
    """Emit a DeprecationWarning pointing users to the new module location.

    Args:
        old: Fully-qualified name of the deprecated module (e.g. ``rfdetr.util.logger``).
        new: Fully-qualified name of the replacement (e.g. ``rfdetr.utilities.logger``).
        deprecated_in: Version where the module was deprecated (full semver).
        remove_in: Version where the module will be removed (full semver).
    """
    warnings.warn(
        f"{old} is deprecated since v{deprecated_in} and will be removed in v{remove_in}; use {new} instead.",
        DeprecationWarning,
        stacklevel=3,
    )

# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Backward-compatibility shim — rfdetr.deploy is deprecated; use rfdetr.export."""

import sys

from mmdet.models.layers.rf_detr.utilities.decorators import _warn_deprecated_module

_warn_deprecated_module("rfdetr.deploy", "rfdetr.export", deprecated_in="1.6.0", remove_in="1.9.0")

# Make old submodule paths still importable without submodule files
import mmdet.models.layers.rf_detr.export.benchmark as _benchmark  # noqa: E402
import mmdet.models.layers.rf_detr.export.main as _export_main  # noqa: E402
from mmdet.models.layers.rf_detr.export import *  # noqa: F403, E402

sys.modules.setdefault("rfdetr.deploy.benchmark", _benchmark)
sys.modules.setdefault("rfdetr.deploy.export", _export_main)

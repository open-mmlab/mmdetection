# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Deprecated: use ``rfdetr.utilities`` or ``rfdetr.training.model_ema`` instead."""

from mmdet.models.layers.rf_detr.utilities.decorators import _warn_deprecated_module

_warn_deprecated_module("rfdetr.util.utils", "rfdetr.utilities", deprecated_in="1.6.0", remove_in="1.9.0")

# Re-export from new locations.
from mmdet.models.layers.rf_detr.training.model_ema import BestMetricHolder, BestMetricSingle, ModelEma  # noqa: F401, E402
from mmdet.models.layers.rf_detr.utilities.reproducibility import seed_all  # noqa: F401, E402
from mmdet.models.layers.rf_detr.utilities.state_dict import clean_state_dict  # noqa: F401, E402

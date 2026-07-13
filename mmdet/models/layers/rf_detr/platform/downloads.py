# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

# The availability check here avoids importing `rfdetr_plus` when the optional
# dependency is not installed. Nested import-time dependencies continue to
# propagate from inside this guarded path.
from mmdet.models.layers.rf_detr.platform import _IS_RFDETR_PLUS_AVAILABLE

if _IS_RFDETR_PLUS_AVAILABLE:
    try:
        from rfdetr_plus.models import downloads as _downloads

        if hasattr(_downloads, "_PLATFORM_MODELS"):
            PLATFORM_MODELS = _downloads._PLATFORM_MODELS
        else:
            PLATFORM_MODELS = _downloads.PLATFORM_MODELS
    except ModuleNotFoundError as ex:
        missing_name = getattr(ex, "name", "")
        if missing_name in {"rfdetr_plus", "rfdetr_plus.models", "rfdetr_plus.models.downloads"}:
            PLATFORM_MODELS = {}
        else:
            raise
else:
    PLATFORM_MODELS = {}

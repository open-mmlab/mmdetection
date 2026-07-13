# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import importlib.util
import warnings

_INSTALL_MSG = (
    "The {name} requires the 'plus' extras for the 'rfdetr' package."
    " Install it with `pip install rfdetr[plus]` (or `pip install rfdetr_plus` if supported)."
)

try:
    _IS_RFDETR_PLUS_AVAILABLE = importlib.util.find_spec("rfdetr_plus") is not None
except ImportError:
    _IS_RFDETR_PLUS_AVAILABLE = False
if not _IS_RFDETR_PLUS_AVAILABLE:
    warnings.warn(
        _INSTALL_MSG.format(name="platform model downloads"),
        ImportWarning,
        stacklevel=2,
    )

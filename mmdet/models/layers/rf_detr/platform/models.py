# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from typing import Any

from mmdet.models.layers.rf_detr.platform import _IS_RFDETR_PLUS_AVAILABLE

__all__: list[str] = []

_PLUS_EXPORTS = {
    "RFDETR2XLarge",
    "RFDETRXLarge",
}

if _IS_RFDETR_PLUS_AVAILABLE:
    from rfdetr_plus.models import (
        RFDETR2XLarge,
        RFDETRXLarge,
    )

    __all__ += [
        "RFDETR2XLarge",
        "RFDETRXLarge",
    ]


def __getattr__(name: str) -> Any:
    """Lazy failure for missing plus exports: warn on import, raise on access."""
    # Only intercept plus-only symbols when the extra package is missing.
    if name in _PLUS_EXPORTS and not _IS_RFDETR_PLUS_AVAILABLE:
        from mmdet.models.layers.rf_detr.platform import _INSTALL_MSG

        # Surface a clear install hint when someone explicitly requests a plus symbol.
        raise ImportError(_INSTALL_MSG.format(name="platform model downloads"))

    # Fall back to the normal attribute lookup error for everything else.
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Copyright (c) OpenMMLab. All rights reserved.
# Vendored RF-DETR core components are Apache-2.0 licensed; see the
# repository-level LICENSE and reports/final_report.md for provenance.
"""Vendored RF-DETR model components for native MMDetection integration.

This package intentionally exposes only the pieces needed by the MMDetection
detector wrapper.  It does not import RF-DETR's public trainer/API facade and
does not depend on the separately cloned ``rf-detr`` repository at runtime.
"""

from .config import RFDETRSmallConfig, TrainConfig
from .inference import ModelContext

__all__ = ['ModelContext', 'RFDETRSmallConfig', 'TrainConfig']

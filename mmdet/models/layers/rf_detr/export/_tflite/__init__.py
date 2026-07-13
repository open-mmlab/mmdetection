# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""TFLite export: ONNX → TFLite conversion via onnx2tf."""

from mmdet.models.layers.rf_detr.export._tflite.converter import _check_onnx2tf_available, export_tflite

try:
    _check_onnx2tf_available()
    _IS_ONNX2TF_AVAILABLE: bool = True
except ImportError:
    _IS_ONNX2TF_AVAILABLE = False

__all__ = ["export_tflite", "_IS_ONNX2TF_AVAILABLE"]

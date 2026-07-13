# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
from collections.abc import Callable
from typing import Any, TypeAlias

OptimizerCallback: TypeAlias = Callable[[Any], Any]


class CustomOpSymbolicRegistry:
    """Registry for custom ONNX symbolic optimizer callbacks."""

    # _SYMBOLICS = {}
    _OPTIMIZER: list[OptimizerCallback] = []

    @classmethod
    def optimizer(cls, fn: OptimizerCallback) -> None:
        cls._OPTIMIZER.append(fn)


def register_optimizer() -> Callable[[OptimizerCallback], OptimizerCallback]:
    def optimizer_wrapper(fn: OptimizerCallback) -> OptimizerCallback:
        CustomOpSymbolicRegistry.optimizer(fn)
        return fn

    return optimizer_wrapper

# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

from typing import Any

from torch import nn


class BackboneBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def get_named_param_lr_pairs(self, args: Any, prefix: str) -> dict[str, dict[str, Any]]:
        raise NotImplementedError

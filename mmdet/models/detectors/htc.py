# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from .cascade_rcnn import CascadeRCNN


@MODELS.register_module()
class HybridTaskCascade(CascadeRCNN):
    """Implementation of `HTC <https://arxiv.org/abs/1901.07518>`_"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @property
    def with_semantic(self) -> bool:
        """bool: whether the detector has a semantic head"""
        return self.roi_head.with_semantic

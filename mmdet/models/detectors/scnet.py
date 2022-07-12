# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from .cascade_rcnn import CascadeRCNN


@MODELS.register_module()
class SCNet(CascadeRCNN):
    """Implementation of `SCNet <https://arxiv.org/abs/2012.10150>`_"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

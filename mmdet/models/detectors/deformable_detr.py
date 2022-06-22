# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from .detr import DETR


@MODELS.register_module()
class DeformableDETR(DETR):
    r"""Implementation of `Deformable DETR: Deformable Transformers for
    End-to-End Object Detection <https://arxiv.org/abs/2010.04159>`_"""

    def __init__(self, *args, **kwargs) -> None:
        super(DETR, self).__init__(*args, **kwargs)

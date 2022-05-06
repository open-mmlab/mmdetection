# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from .detr import DETR


@MODELS.register_module()
class DeformableDETR(DETR):

    def __init__(self, *args, **kwargs):
        super(DETR, self).__init__(*args, **kwargs)

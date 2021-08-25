# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .detr import DETR


@DETECTORS.register_module()
class DeformableDETR(DETR):

    def __init__(self, *args, **kwargs):
        super(DETR, self).__init__(*args, **kwargs)

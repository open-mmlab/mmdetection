# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base

with read_base():
    from .._base_.models.faster_rcnn_r50_fpn import *
    from ..common.ms_3x_coco import *

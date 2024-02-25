# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base
from mmengine.model.weight_init import PretrainedInit

with read_base():
    from .mask2former_r50_8xb2_8e_youtubevis2019 import *

model.update(
    dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type=PretrainedInit, checkpoint='torchvision://resnet101')),
        init_cfg=dict(
            type=PretrainedInit,
            checkpoint='https://download.openmmlab.com/mmdetection/v3.0/'
            'mask2former/mask2former_r101_8xb2-lsj-50e_coco/'
            'mask2former_r101_8xb2-lsj-50e_coco_20220426_100250-ecf181e2.pth'))
)

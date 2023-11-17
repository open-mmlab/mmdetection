# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base
from mmengine.model.weight_init import PretrainedInit

with read_base():
    from .faster_rcnn_r50_caffe_fpn_ms_1x_coco import *

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=500),  # noqa
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[16, 23],
        gamma=0.1)
]

train_cfg.update(dict(max_epochs=24))

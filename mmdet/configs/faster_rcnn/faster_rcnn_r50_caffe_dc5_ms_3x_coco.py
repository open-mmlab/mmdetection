# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

with read_base():
    from .faster_rcnn_r50_caffe_dc5_1x_coco import *

train_cfg.update(dict(max_epochs=36))

param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0,
         end=500),  # noqa
    dict(
        type=MultiStepLR,
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[28, 34],
        gamma=0.1)
]

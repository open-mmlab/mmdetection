# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from mmengine.runner import IterBasedTrainLoop

with read_base():
    from .faster_rcnn_r50_caffe_fpn_1x_coco import *

max_iter = 90000

param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[60000, 80000],
        gamma=0.1)
]

train_cfg.update(
    dict(
        _delete_=True,
        type=IterBasedTrainLoop,
        max_iters=max_iter,
        val_interval=10000))
default_hooks.update(dict(checkpoint=dict(by_epoch=False, interval=10000)))
log_processor.update(dict(by_epoch=False))

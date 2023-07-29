# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .htc_r50_fpn_1x_coco import *

from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

from mmdet.models.backbones.resnext import ResNeXt

model.update(
    dict(
        backbone=dict(
            type=ResNeXt,
            depth=101,
            groups=32,
            base_width=4,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type=BatchNorm2d, requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type=PretrainedInit,
                checkpoint='open-mmlab://resnext101_32x4d'))))

train_dataloader.update(dict(batch_size=1, num_workers=1))

# learning policy
max_epochs = 20
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[16, 19],
        gamma=0.1)
]
train_cfg.update(dict(max_epochs=max_epochs))

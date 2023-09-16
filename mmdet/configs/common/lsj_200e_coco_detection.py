# Copyright (c) OpenMMLab. All rights reserved.

# Please refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta for more details. # noqa
# mmcv >= 2.0.1
# mmengine >= 0.8.0

from mmengine.config import read_base

with read_base():
    from .lsj_100e_coco_detection import *

# 8x25=200e
train_dataloader.update(dict(dataset=dict(times=8)))

# learning rate
param_scheduler = [
    dict(type=LinearLR, start_factor=0.067, by_epoch=False, begin=0, end=1000),
    dict(
        type=MultiStepLR,
        begin=0,
        end=25,
        by_epoch=True,
        milestones=[22, 24],
        gamma=0.1)
]

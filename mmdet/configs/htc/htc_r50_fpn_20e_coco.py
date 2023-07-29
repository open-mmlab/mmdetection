# Copyright (c) OpenMMLab. All rights reserved.

# Please refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta for more details. # noqa
# mmcv >= 2.0.1
# mmengine >= 0.8.0
from mmengine.config import read_base

with read_base():
    from .htc_r50_fpn_1x_coco import *

from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

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

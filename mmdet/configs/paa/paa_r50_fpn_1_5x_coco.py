if '_base_':
    from .paa_r50_fpn_1x_coco import *
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

max_epochs = 18

# learning rate
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[12, 16],
        gamma=0.1)
]

# training schedule for 1.5x
train_cfg.merge(dict(max_epochs=max_epochs))

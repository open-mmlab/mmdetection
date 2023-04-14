if '_base_':
    from .reppoints_moment_r50_fpn_gn_head_gn_1x_coco import *
from mmengine.runner.loops import EpochBasedTrainLoop
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

max_epochs = 24

train_cfg.merge(
    dict(type=EpochBasedTrainLoop, max_epochs=max_epochs, val_interval=1))
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

if '_base_':
    from .dino_4scale_r50_8xb2_12e_coco import *
from mmengine.runner.loops import EpochBasedTrainLoop
from mmengine.optim.scheduler.lr_scheduler import MultiStepLR

max_epochs = 24
train_cfg.merge(
    dict(type=EpochBasedTrainLoop, max_epochs=max_epochs, val_interval=1))
param_scheduler = [
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[20],
        gamma=0.1)
]

if '_base_':
    from .dino_5scale_swin_l_8xb2_12e_coco import *
from mmengine.runner.loops import EpochBasedTrainLoop
from mmengine.optim.scheduler.lr_scheduler import MultiStepLR

max_epochs = 36
train_cfg.merge(
    dict(type=EpochBasedTrainLoop, max_epochs=max_epochs, val_interval=1))
param_scheduler = [
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]

from mmengine.config import read_base

with read_base():
    from .dino_5scale_swin_l_8xb2_12e_coco import *

from mmengine.runner.loops import EpochBasedTrainLoop
max_epochs = 36
train_cfg.update(
    dict(
        type=EpochBasedTrainLoop, max_epochs=max_epochs, val_interval=1))
param_scheduler = [
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]

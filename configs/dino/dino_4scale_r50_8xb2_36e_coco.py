from mmengine.config import read_base

with read_base():
    from .dino_4scale_r50_8xb2_12e_coco import *

from mmengine.runner.loops import EpochBasedTrainLoop

max_epochs = 36
train_cfg.update(
    dict(
        type=EpochBasedTrainLoop,
        max_epochs=max_epochs,
        val_interval=1))

param_scheduler[0].update(
    dict(milestones=[30]))

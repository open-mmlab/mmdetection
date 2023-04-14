if '_base_':
    from .detr_r50_8xb2_150e_coco import *
from mmengine.runner.loops import EpochBasedTrainLoop
from mmengine.optim.scheduler.lr_scheduler import MultiStepLR

# learning policy
max_epochs = 500
train_cfg.merge(
    dict(type=EpochBasedTrainLoop, max_epochs=max_epochs, val_interval=10))

param_scheduler = [
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[334],
        gamma=0.1)
]

# only keep latest 2 checkpoints
default_hooks.merge(dict(checkpoint=dict(max_keep_ckpts=2)))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=16))

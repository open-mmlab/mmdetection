if '_base_':
    from .._base_.models.mask_rcnn_r50_fpn import *
    from .._base_.datasets.deepfashion import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
from mmengine.runner.loops import EpochBasedTrainLoop
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

model.merge(
    dict(
        roi_head=dict(
            bbox_head=dict(num_classes=15), mask_head=dict(num_classes=15))))
# runtime settings
max_epochs = 15
train_cfg.merge(
    dict(type=EpochBasedTrainLoop, max_epochs=max_epochs, val_interval=1))
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

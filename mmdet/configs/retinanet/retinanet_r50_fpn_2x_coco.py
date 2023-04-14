if '_base_':
    from .._base_.models.retinanet_r50_fpn import *
    from .._base_.datasets.coco_detection import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from torch.optim.sgd import SGD

# training schedule for 2x
train_cfg.merge(dict(max_epochs=24))

# learning rate policy
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

# optimizer
optim_wrapper.merge(
    dict(optimizer=dict(type=SGD, lr=0.01, momentum=0.9, weight_decay=0.0001)))

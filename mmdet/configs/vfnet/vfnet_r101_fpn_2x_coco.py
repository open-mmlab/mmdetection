if '_base_':
    from .vfnet_r50_fpn_1x_coco import *
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

model.merge(
    dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101'))))
# learning policy
max_epochs = 24
param_scheduler = [
    dict(type=LinearLR, start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

train_cfg.merge(dict(max_epochs=max_epochs))

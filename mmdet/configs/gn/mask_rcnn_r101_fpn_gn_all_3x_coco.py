if '_base_':
    from .mask_rcnn_r101_fpn_gn_all_2x_coco import *
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

# learning policy
max_epochs = 36
train_cfg.merge(dict(max_epochs=max_epochs))

# learning rate
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[28, 34],
        gamma=0.1)
]

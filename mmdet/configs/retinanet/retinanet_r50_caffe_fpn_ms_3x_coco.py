if '_base_':
    from .retinanet_r50_caffe_fpn_ms_1x_coco import *
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

# training schedule for 2x
train_cfg.merge(dict(max_epochs=36))

# learning rate policy
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[28, 34],
        gamma=0.1)
]

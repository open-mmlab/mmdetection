if '_base_':
    from .paa_r50_fpn_1x_coco import *
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmcv.transforms.processing import RandomResize
from mmdet.datasets.transforms.transforms import RandomFlip
from mmdet.datasets.transforms.formatting import PackDetInputs

max_epochs = 36

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

# training schedule for 3x
train_cfg.merge(dict(max_epochs=max_epochs))

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(type=RandomResize, scale=[(1333, 640), (1333, 800)], keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]
train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))

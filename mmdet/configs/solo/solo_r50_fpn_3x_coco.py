if '_base_':
    from .solo_r50_fpn_1x_coco import *
from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmcv.transforms.processing import RandomChoiceResize
from mmdet.datasets.transforms.transforms import RandomFlip
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True, with_mask=True),
    dict(
        type=RandomChoiceResize,
        scales=[(1333, 800), (1333, 768), (1333, 736), (1333, 704),
                (1333, 672), (1333, 640)],
        keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]
train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))

# training schedule for 3x
max_epochs = 36
train_cfg.merge(dict(by_epoch=True, max_epochs=max_epochs))

# learning rate
param_scheduler = [
    dict(
        type=LinearLR, start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]

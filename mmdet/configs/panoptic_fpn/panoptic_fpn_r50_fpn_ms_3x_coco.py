if '_base_':
    from .panoptic_fpn_r50_fpn_1x_coco import *
from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadPanopticAnnotations
from mmcv.transforms.processing import RandomResize
from mmdet.datasets.transforms.transforms import RandomFlip
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

# In mstrain 3x config, img_scale=[(1333, 640), (1333, 800)],
# multiscale_mode='range'
train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(
        type=LoadPanopticAnnotations,
        with_bbox=True,
        with_mask=True,
        with_seg=True),
    dict(type=RandomResize, scale=[(1333, 640), (1333, 800)], keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]
train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))

# TODO: Use RepeatDataset to speed up training
# training schedule for 3x
train_cfg.merge(dict(max_epochs=36, val_interval=3))

# learning rate
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[24, 33],
        gamma=0.1)
]

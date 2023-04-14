if '_base_':
    from .fcos_hrnetv2p_w32_gn_head_4xb4_1x_coco import *
from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmcv.transforms.processing import RandomChoiceResize
from mmdet.datasets.transforms.transforms import RandomFlip
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

model.merge(
    dict(
        data_preprocessor=dict(
            mean=[103.53, 116.28, 123.675],
            std=[57.375, 57.12, 58.395],
            bgr_to_rgb=False)))

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=RandomChoiceResize,
        scales=[(1333, 640), (1333, 800)],
        keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]

train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))

# learning policy
max_epochs = 24
train_cfg.merge(dict(max_epochs=max_epochs))
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

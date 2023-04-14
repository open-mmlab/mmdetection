if '_base_':
    from .fcos_r50_caffe_fpn_gn_head_1x_coco import *
from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor
from mmdet.models.backbones.resnext import ResNeXt
from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmcv.transforms.processing import RandomChoiceResize
from mmdet.datasets.transforms.transforms import RandomFlip
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmengine.optim.scheduler.lr_scheduler import ConstantLR, MultiStepLR

# model settings
model.merge(
    dict(
        data_preprocessor=dict(
            type=DetDataPreprocessor,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=32),
        backbone=dict(
            type=ResNeXt,
            depth=101,
            groups=64,
            base_width=4,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://resnext101_64x4d'))))

# dataset settings
train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=RandomChoiceResize,
        scale=[(1333, 640), (1333, 800)],
        keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]
train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))

# training schedule for 2x
max_epochs = 24
train_cfg.merge(dict(max_epochs=max_epochs))

# learning rate
param_scheduler = [
    dict(type=ConstantLR, factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

# Copyright (c) OpenMMLab. All rights reserved.

# Please refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta for more details. # noqa
# mmcv >= 2.0.1
# mmengine >= 0.8.0

from mmengine.config import read_base

with read_base():
    from .rtmdet_l_8xb32_300e_coco import *

from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms.processing import RandomResize
from mmengine.hooks.ema_hook import EMAHook

from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.datasets.transforms.transforms import (CachedMixUp, CachedMosaic,
                                                  Pad, RandomCrop, RandomFlip,
                                                  Resize, YOLOXHSVRandomAug)
from mmdet.engine.hooks.pipeline_switch_hook import PipelineSwitchHook
from mmdet.models.layers.ema import ExpMomentumEMA

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth'  # noqa
model.update(
    dict(
        backbone=dict(
            deepen_factor=0.33,
            widen_factor=0.5,
            init_cfg=dict(
                type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
        neck=dict(
            in_channels=[128, 256, 512], out_channels=128, num_csp_blocks=1),
        bbox_head=dict(in_channels=128, feat_channels=128, exp_on_reg=False)))

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(type=CachedMosaic, img_scale=(640, 640), pad_val=114.0),
    dict(
        type=RandomResize,
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        resize_type=Resize,
        keep_ratio=True),
    dict(type=RandomCrop, crop_size=(640, 640)),
    dict(type=YOLOXHSVRandomAug),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type=CachedMixUp,
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type=PackDetInputs)
]

train_pipeline_stage2 = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=RandomResize,
        scale=(640, 640),
        ratio_range=(0.5, 2.0),
        resize_type=Resize,
        keep_ratio=True),
    dict(type=RandomCrop, crop_size=(640, 640)),
    dict(type=YOLOXHSVRandomAug),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type=PackDetInputs)
]

train_dataloader.update(dict(dataset=dict(pipeline=train_pipeline)))

custom_hooks = [
    dict(
        type=EMAHook,
        ema_type=ExpMomentumEMA,
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type=PipelineSwitchHook,
        switch_epoch=280,
        switch_pipeline=train_pipeline_stage2)
]

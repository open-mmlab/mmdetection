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
from torch.nn.modules.activation import SiLU

from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.datasets.transforms.loading import (FilterAnnotations,
                                               LoadAnnotations)
from mmdet.datasets.transforms.transforms import (CachedMixUp, CachedMosaic,
                                                  Pad, RandomCrop, RandomFlip,
                                                  Resize, YOLOXHSVRandomAug)
from mmdet.engine.hooks.pipeline_switch_hook import PipelineSwitchHook
from mmdet.models.dense_heads.rtmdet_ins_head import RTMDetInsSepBNHead
from mmdet.models.layers.ema import ExpMomentumEMA
from mmdet.models.losses.dice_loss import DiceLoss
from mmdet.models.losses.gfocal_loss import QualityFocalLoss
from mmdet.models.losses.iou_loss import GIoULoss
from mmdet.models.task_modules.coders.distance_point_bbox_coder import \
    DistancePointBBoxCoder
from mmdet.models.task_modules.prior_generators.point_generator import \
    MlvlPointGenerator

model.merge(
    dict(
        bbox_head=dict(
            _delete_=True,
            type=RTMDetInsSepBNHead,
            num_classes=80,
            in_channels=256,
            stacked_convs=2,
            share_conv=True,
            pred_kernel_size=1,
            feat_channels=256,
            act_cfg=dict(type=SiLU, inplace=True),
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            anchor_generator=dict(
                type=MlvlPointGenerator, offset=0, strides=[8, 16, 32]),
            bbox_coder=dict(type=DistancePointBBoxCoder),
            loss_cls=dict(
                type=QualityFocalLoss,
                use_sigmoid=True,
                beta=2.0,
                loss_weight=1.0),
            loss_bbox=dict(type=GIoULoss, loss_weight=2.0),
            loss_mask=dict(
                type=DiceLoss, loss_weight=2.0, eps=5e-6, reduction='mean')),
        test_cfg=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100,
            mask_thr_binary=0.5),
    ))

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(
        type=LoadAnnotations, with_bbox=True, with_mask=True, poly2mask=False),
    dict(type=CachedMosaic, img_scale=(640, 640), pad_val=114.0),
    dict(
        type=RandomResize,
        scale=(1280, 1280),
        ratio_range=(0.1, 2.0),
        resize_type=Resize,
        keep_ratio=True),
    dict(
        type=RandomCrop,
        crop_size=(640, 640),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type=YOLOXHSVRandomAug),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type=CachedMixUp,
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type=FilterAnnotations, min_gt_bbox_wh=(1, 1)),
    dict(type=PackDetInputs)
]

train_dataloader.update(
    dict(pin_memory=True, dataset=dict(pipeline=train_pipeline)))

train_pipeline_stage2 = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(
        type=LoadAnnotations, with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type=RandomResize,
        scale=(640, 640),
        ratio_range=(0.1, 2.0),
        resize_type=Resize,
        keep_ratio=True),
    dict(
        type=RandomCrop,
        crop_size=(640, 640),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type=FilterAnnotations, min_gt_bbox_wh=(1, 1)),
    dict(type=YOLOXHSVRandomAug),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type=PackDetInputs)
]
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

val_evaluator.update(dict(metric=['bbox', 'segm']))
test_evaluator = val_evaluator

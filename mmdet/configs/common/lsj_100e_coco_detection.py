# Copyright (c) OpenMMLab. All rights reserved.

# Please refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta for more details. # noqa
# mmcv >= 2.0.1
# mmengine >= 0.8.0

from mmengine.config import read_base

with read_base():
    from .._base_.default_runtime import *

from mmengine.dataset.sampler import DefaultSampler
from mmengine.optim import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from mmengine.runner.loops import EpochBasedTrainLoop, TestLoop, ValLoop
from torch.optim import SGD

from mmdet.datasets import CocoDataset, RepeatDataset
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.datasets.transforms.loading import (FilterAnnotations,
                                               LoadAnnotations,
                                               LoadImageFromFile)
from mmdet.datasets.transforms.transforms import (CachedMixUp, CachedMosaic,
                                                  Pad, RandomCrop, RandomFlip,
                                                  RandomResize, Resize)
from mmdet.evaluation import CocoMetric

# dataset settings
dataset_type = CocoDataset
data_root = 'data/coco/'
image_size = (1024, 1024)

backend_args = None

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True, with_mask=True),
    dict(
        type=RandomResize,
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type=RandomCrop,
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type=FilterAnnotations, min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]
test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# Use RepeatDataset to speed up training
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=RepeatDataset,
        times=4,  # simply change this from 2 to 16 for 50e - 400e training.
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='annotations/instances_train2017.json',
            data_prefix=dict(img='train2017/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            backend_args=backend_args)))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type=CocoMetric,
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

max_epochs = 25

train_cfg = dict(
    type=EpochBasedTrainLoop, max_epochs=max_epochs, val_interval=5)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

# optimizer assumes bs=64
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=SGD, lr=0.1, momentum=0.9, weight_decay=0.00004))

# learning rate
param_scheduler = [
    dict(type=LinearLR, start_factor=0.067, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[22, 24],
        gamma=0.1)
]

# only keep latest 2 checkpoints
default_hooks.update(dict(checkpoint=dict(max_keep_ckpts=2)))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (32 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)

#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

_base_ = '../_base_/default_runtime.py'

# model settings
model = dict(
    type='YOLOX',
    backbone=dict(
        type='CSPDarknet',
        dep_mul=0.33,
        wid_mul=0.50,
        out_features=('dark3', 'dark4', 'dark5')),
    neck=dict(
        type='YOLOPAFPN',
        depth=0.33,
        width=0.50,
        in_features=('dark3', 'dark4', 'dark5'),
        in_channels=[256, 512, 1024]),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=80,
        in_channels=[256, 512, 1024],
        width=0.50,
        strides=[8, 16, 32]),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=100))

# TODO transform
# random_size = (14, 26)
# degrees = 10.0
# translate = 0.1
# scale = (0.1, 2)
# mscale = (0.8, 1.6)
# shear = 2.0
# perspective = 0.0
# enable_mixup = True

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(640, 640)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(640, 640)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric=['bbox'])
cudnn_benchmark = True

# optimizer
optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',  # TODO 'yoloxwarmcos'
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.,  # TODO check whether the same as `warmup_lr = 0`
    warmup_by_epoch=True,
    min_lr_ratio=0.05)
runner = dict(type='EpochBasedRunner', max_epochs=300)

# no_aug_epochs = 15  # TODO
# ema = True  # TODO

# TODO
# https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/exp/yolox_base.py

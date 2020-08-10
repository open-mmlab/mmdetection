_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py', '../_base_/default_runtime.py'
]
cudnn_benchmark = True
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        with_cp=True,
        frozen_stages=1,
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        relu_before_extra_convs=True,
        no_norm_on_lateral=True,
        norm_cfg=norm_cfg),
    bbox_head=dict(type='RetinaSepBNHead', num_ins=5, norm_cfg=norm_cfg))
# training and testing settings
train_cfg = dict(assigner=dict(neg_iou_thr=0.5))
# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(
        type='Mosaic',
        sub_pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Expand',
                mean=img_norm_cfg['mean'],
                to_rgb=img_norm_cfg['to_rgb'],
                ratio_range=(1.4, 1.4),
                prob=1.0),
            dict(
                type='RandomCrop',
                crop_size=None,
                min_crop_size=0.4286,  # 0.6 / 1.4
                allow_negative_crop=True),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=False)
        ],
        size=(640, 640),
        min_offset=0.2),
    dict(type='Normalize', **img_norm_cfg),
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
            dict(type='Pad', size_divisor=64),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(pipeline=train_pipeline, num_samples_per_iter=4),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.08,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(norm_decay_mult=0, bypass_duplicate=True))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    step=[30, 40])
# runtime settings
total_epochs = 50

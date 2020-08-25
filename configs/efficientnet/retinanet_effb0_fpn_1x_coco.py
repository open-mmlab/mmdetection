_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
model = dict(
    pretrained='open-mmlab://efficientnet_b0',
    backbone=dict(
        _delete_=True, type='EfficientNet', scale=0, frozen_stages=1),
    neck=dict(
        type='FPN',
        in_channels=[40, 112, 320],
        start_level=0,
        out_channels=256,
        num_outs=5))
img_norm_cfg = dict(
    # The mean and std is used in PyCls
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Resize',
        img_scale=(512, 512),
        keep_ratio=True,
        ratio_range=(0.1, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), allow_negative_crop=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=3,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# lr = 0.08 * (batch_size / 64)
optimizer = dict(type='SGD', lr=0.08, momentum=0.9, weight_decay=4e-5)

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr=0,
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=0.1,
    warmup_by_epoch=True)

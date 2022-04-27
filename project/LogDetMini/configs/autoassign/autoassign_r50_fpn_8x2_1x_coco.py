# We follow the original implementation which
# adopts the Caffe pre-trained backbone.
_base_ = [
    '../_base_/datasets.py',
    '../_base_/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='AutoAssign',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='GN', num_groups=32,requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_gn')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5,
        relu_before_extra_convs=True,
        #init_cfg=dict(type='Caffe2Xavier', layer='Conv2d')
        ),
    bbox_head=dict(
        type='AutoAssignHead',
        num_classes=50,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_bbox=dict(type='GIoULoss', loss_weight=5.0)),
    train_cfg=None,
    test_cfg=dict(
        nms_pre=500,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type="RandomCrop", crop_size=(800, 800), crop_type='absolute'),
    dict(type='Resize', img_scale=(800, 800), ratio_range=(0.3, 1.5), keep_ratio=True),
    dict(type='Rotate', level=1, max_rotate_angle=30, prob=0.5),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1200, 1200),(2048, 2048)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(lr=0.0025, paramwise_cfg=dict(norm_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
load_from = '/mnt/d/Github/mmdetection/project/LogDetMini/workdir/autoassign/baseline_2x/latest.pth'
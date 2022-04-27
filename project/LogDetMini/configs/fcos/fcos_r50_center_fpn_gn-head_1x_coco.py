_base_ = [
    '../_base_/datasets.py',
    '../_base_/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='FCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=50,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        norm_on_bbox=True,
        centerness_on_reg=True,
        center_sampling=True,
        center_sample_radius=1.5,
        conv_bias=True,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=500,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type="RandomCrop", crop_size=(800, 800), crop_type='absolute'),
    dict(type='Resize', img_scale=(800, 800), ratio_range=(0.8, 1.2), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
#dict(type='Rotate', level=1, max_rotate_angle=30, prob=0.6),
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333,800),
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
#fp16 = dict(loss_scale='dynamic')  
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    lr=0.00375, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))  #学习率的设置尤为关键：lr = 0.00125*batch_size
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
load_from = '/mnt/d/Github/mmdetection/project/LogDetMini/workdir/fcos/baseline3/epoch_12.pth'
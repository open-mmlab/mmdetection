_base_ = [
    '../_base_/datasets/wider_face.py',
    '../_base_/default_runtime.py',
]

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomSquareCrop', crop_ratio_range=[0.3, 1.0]),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
        saturate=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32, pad_val=0),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(500, 750), (800, 1200), (1100, 1650), (1400, 2100),
                   (1700, 2550)],  # (1100, 1650) for single scale
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32, pad_val=0),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=8,
    train=dict(
        times=1, dataset=dict(min_size=1, offset=0, pipeline=train_pipeline)),
    val=dict(min_size=1, offset=0, pipeline=val_pipeline),
    test=dict(min_size=1, offset=0, pipeline=test_pipeline))

model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type='ResNet',
        depth=152,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN'),
        norm_eval=False,
        dcn=None,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet152')),
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=0,
            add_extra_convs='on_input',
            num_outs=5,
            norm_cfg=dict(type='BN'),
            upsample_cfg=dict(mode='bilinear')),
        dict(
            type='SSHContext',
            in_channel=256,
            out_channel=256,
            norm_cfg=dict(type='BN')),
    ],
    bbox_head=dict(
        type='RetinaFaceHead',
        in_channels=256,
        num_classes=1,
        stacked_convs=0,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=1,
            scales_per_octave=3,
            ratios=[1.0],
            strides=[4, 8, 16, 32, 64],
            base_sizes=[16, 32, 64, 128, 256]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=3.0, loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.3,
            min_pos_iou=0.,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=-1,
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=-1))

evaluation = dict(interval=5)
optimizer = dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()

max_epochs = 80
lr_config = dict(
    policy='Step',
    step=[55, 68],
    gamma=0.1,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.1,
    warmup_by_epoch=True)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
checkpoint_config = dict(interval=10)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None

workflow = [('train', 1)]

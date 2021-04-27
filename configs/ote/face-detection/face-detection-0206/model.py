# model settings
input_size = 640
width_mult = 1.0
model = dict(
    type='ATSS',
    backbone=dict(
        type='resnet152b',
        out_indices=(1, 2, 3, 4),
        frozen_stages=-1,
        norm_eval=False,
        pretrained=True,
    ),
    neck=dict(
        type='RSSH_FPN',
        in_channels=[int(width_mult * 256),
                     int(width_mult * 512),
                     int(width_mult * 1024),
                     int(width_mult * 2048),
                     ],
        out_channels=256,
        start_level=0,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='ATSSHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=128,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.5),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=750))
cudnn_benchmark = True
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/WIDERFace/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.1),
    dict(type='Resize', img_scale=(input_size, input_size), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(input_size, input_size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=3,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            classes=('face',),
            ann_file=data_root + '/train.json',
            min_size=0,
            img_prefix=data_root,
            pipeline=train_pipeline
        )
    ),
    val=dict(
        type=dataset_type,
        classes=('face',),
        ann_file=data_root + '/val.json',
        img_prefix=data_root,
        test_mode=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=('face',),
        ann_file=data_root + '/val.json',
        img_prefix=data_root,
        test_mode=True,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1200,
    warmup_ratio=1.0 / 3,
    step=[40, 55, 65])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 70
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'outputs/face-detection-0206'
load_from = None
resume_from = None
workflow = [('train', 1)]

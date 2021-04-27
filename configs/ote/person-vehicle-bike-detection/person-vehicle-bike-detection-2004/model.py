# model settings
model = dict(
    type='ATSS',
    backbone=dict(
        type='mobilenetv2_w1',
        out_indices=(2, 3, 4, 5),
        frozen_stages=-1,
        norm_eval=False,
        pretrained=True
    ),
    neck=dict(
        type='FPN',
        in_channels=[24, 32, 96, 320],
        out_channels=32,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,
        num_outs=5),
    bbox_head=dict(
        type='ATSSHead',
        num_classes=3,
        in_channels=32,
        stacked_convs=4,
        feat_channels=32,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[0.5, 1.0, 2.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
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
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
# dataset settings
dataset_type = 'CocoDataset'
data_root = '../../data/airport/'
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='Expand', ratio_range=(1, 3)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(
        type='Resize',
        img_scale=[(448, 256), (448, 320)],
        keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(448, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=54,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            classes=('vehicle', 'person', 'non-vehicle'),
            ann_file=data_root + 'annotation_example_train.json',
            img_prefix=data_root + 'train',
            pipeline=train_pipeline
        )
    ),
    val=dict(
        type=dataset_type,
        classes=('vehicle', 'person', 'non-vehicle'),
        ann_file=data_root + 'annotation_example_val.json',
        img_prefix=data_root + 'val',
        test_mode=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=('vehicle', 'person', 'non-vehicle'),
        ann_file=data_root + 'annotation_example_val.json',
        img_prefix=data_root + 'val',
        test_mode=True,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.2,
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[10, 15, 18])
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
total_epochs = 20
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'outputs/person-vehicle-bike-detection-2004'
load_from = None
resume_from = None
workflow = [('train', 1)]

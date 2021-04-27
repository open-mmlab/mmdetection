# model settings
model = dict(
    type='FCOS',
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
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=1,
        in_channels=32,
        stacked_convs=4,
        feat_channels=32,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
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
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.25,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
evaluation = dict(interval=1, metric=['bbox', 'f1'], score_thr=0.25)
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/text-dataset/'
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(
        type='Resize',
        img_scale=[(704, 704), (844, 704), (704, 844), (564, 704), (704, 564)],
        multiscale_mode='value',
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
        img_scale=(704, 704),
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
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            classes=('text',),
            ann_file=data_root + 'IC13TRAIN_IC15_IC17_IC19_MSRATD500_COCOTEXT.json',
            img_prefix=data_root,
            pipeline=train_pipeline),
    ),
    val=dict(
        type=dataset_type,
        classes=('text',),
        ann_file=data_root + 'IC13TEST.json',
        img_prefix=data_root,
        test_mode=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=('text',),
        ann_file=data_root + 'IC13TEST.json',
        img_prefix=data_root,
        test_mode=True,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[14, 22])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 25
device_ids = range(4)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'outputs/horizontal-text-detection'
load_from = None
resume_from = None
workflow = [('train', 1)]

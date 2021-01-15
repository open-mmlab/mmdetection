# model settings
input_size = 512
indices = (3, 4, 5)
channels = 64
num_levels = 5
act_cfg = 'silu'
separable_conv = True
num_epochs = 300
delta = 0.1

model = dict(
    type='RetinaNet',
    backbone=dict(
        type='efficientnet_b0',
        out_indices=indices,
        frozen_stages=-1,
        norm_eval=False,
        pretrained=True,
        verbose=False,
    ),
    neck=dict(
        type='BiFPN',
        in_channels=[40, 112, 320],
        out_channels=channels,
        input_indices=indices,
        num_outs=num_levels,
        strides=[8, 16, 32],
        num_layers=3,
        weight_method='fast_attn',
        act_cfg=act_cfg,
        separable_conv=separable_conv,
        epsilon=0.0001
    ),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=channels,
        stacked_convs=3,
        feat_channels=channels,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.5,
            alpha=0.25,
            loss_weight=1.0),
        # Replicates huber loss
        loss_bbox=dict(type='SmoothL1Loss', beta=delta, loss_weight=delta)
    ))
# training and testing settings
cudnn_benchmark = True
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# model training and testing settings
# dataset settings
dataset_type = 'CocoDataset'
data_root = './data/coco/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.374], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=(input_size, input_size),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(input_size, input_size)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(input_size, input_size)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
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
            dict(type='RandomFlip'),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=20,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017',
            pipeline=train_pipeline
        )
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017',
        test_mode=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017',
        test_mode=True,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.08, momentum=0.9, weight_decay=0.00004)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineRestart',
    periods=[num_epochs],
    min_lr=0.00001,
    min_lr_ratio=None,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.0001,
    warmup_by_epoch=True
)
checkpoint_config = dict(interval=5)
# yapf:disable
log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
# runtime settings
total_epochs = num_epochs
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'outputs/retinanet_effd0_bifpn_1x_coco'
load_from = None
resume_from = None
workflow = [('train', 1)]

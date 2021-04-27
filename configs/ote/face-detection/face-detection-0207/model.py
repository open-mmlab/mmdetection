# model settings
input_size = 448
width_mult = 1.0
model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type='shufflenetv2_w1',
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_eval=False,
        pretrained=True
    ),
    neck=None,
    bbox_head=dict(
        type='SSDHead',
        num_classes=1,
        in_channels=(int(width_mult * 232), int(width_mult * 464)),
        anchor_generator=dict(
            type='SSDAnchorGeneratorClustered',
            strides=(16, 32),
            widths=(
                [14.0373, 37.4827, 21.952, 51.8187],
                [213.5467, 115.584, 192.3413, 76.3093, 112.896]),
            heights=(
                [22.4, 59.136, 38.08, 94.3787],
                [339.7333, 243.264, 185.92, 156.9493, 108.416]),
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=(.0, .0, .0, .0),
            target_stds=(0.1, 0.1, 0.2, 0.2),),
        depthwise_heads=True,
        depthwise_heads_activations='relu',
        loss_balancing=True),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.4,
            neg_iou_thr=0.4,
            min_pos_iou=0.,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        smoothl1_beta=1.,
        use_giou=False,
        use_focal=False,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200))
cudnn_benchmark = True
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/WIDERFace/'
img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
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
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            classes=('face',),
            ann_file=data_root + '/train.json',
            min_size=17,
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
optimizer_config = dict()
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
work_dir = 'output/face-detection-0207'
load_from = None
resume_from = None
workflow = [('train', 1)]

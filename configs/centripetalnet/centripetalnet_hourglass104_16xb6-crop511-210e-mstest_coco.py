_base_ = [
    '../_base_/default_runtime.py', '../_base_/datasets/coco_detection.py'
]

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True)

# model settings
model = dict(
    type='CornerNet',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='HourglassNet',
        downsample_times=5,
        num_stacks=2,
        stage_channels=[256, 256, 384, 384, 384, 512],
        stage_blocks=[2, 2, 2, 2, 2, 4],
        norm_cfg=dict(type='BN', requires_grad=True)),
    neck=None,
    bbox_head=dict(
        type='CentripetalHead',
        num_classes=80,
        in_channels=256,
        num_feat_levels=2,
        corner_emb_channels=0,
        loss_heatmap=dict(
            type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=1),
        loss_offset=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1),
        loss_guiding_shift=dict(
            type='SmoothL1Loss', beta=1.0, loss_weight=0.05),
        loss_centripetal_shift=dict(
            type='SmoothL1Loss', beta=1.0, loss_weight=1)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        corner_topk=100,
        local_maximum_kernel=3,
        distance_threshold=0.5,
        score_thr=0.05,
        max_per_img=100,
        nms=dict(type='soft_nms', iou_threshold=0.5, method='gaussian')))

# data settings
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        # The cropped images are padded into squares during training,
        # but may be smaller than crop_size.
        type='RandomCenterCropPad',
        crop_size=(511, 511),
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        test_mode=False,
        test_pad_mode=None,
        mean=data_preprocessor['mean'],
        std=data_preprocessor['std'],
        # Image data is not converted to rgb.
        to_rgb=data_preprocessor['bgr_to_rgb']),
    dict(type='Resize', scale=(511, 511), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args=_base_.backend_args),
    # don't need Resize
    dict(
        type='RandomCenterCropPad',
        crop_size=None,
        ratios=None,
        border=None,
        test_mode=True,
        test_pad_mode=['logical_or', 127],
        mean=data_preprocessor['mean'],
        std=data_preprocessor['std'],
        # Image data is not converted to rgb.
        to_rgb=data_preprocessor['bgr_to_rgb']),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'border'))
]

train_dataloader = dict(
    batch_size=6,
    num_workers=3,
    batch_sampler=None,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.0005),
    clip_grad=dict(max_norm=35, norm_type=2))

max_epochs = 210

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[190],
        gamma=0.1)
]

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (6 samples per GPU)
auto_scale_lr = dict(base_batch_size=96)

tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(
        nms=dict(type='soft_nms', iou_threshold=0.5, method='gaussian'),
        max_per_img=100))

tta_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args=_base_.backend_args),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                # ``RandomFlip`` must be placed before ``RandomCenterCropPad``,
                # otherwise bounding box coordinates after flipping cannot be
                # recovered correctly.
                dict(type='RandomFlip', prob=1.),
                dict(type='RandomFlip', prob=0.)
            ],
            [
                dict(
                    type='RandomCenterCropPad',
                    crop_size=None,
                    ratios=None,
                    border=None,
                    test_mode=True,
                    test_pad_mode=['logical_or', 127],
                    mean=data_preprocessor['mean'],
                    std=data_preprocessor['std'],
                    # Image data is not converted to rgb.
                    to_rgb=data_preprocessor['bgr_to_rgb'])
            ],
            [dict(type='LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'flip', 'flip_direction', 'border'))
            ]
        ])
]

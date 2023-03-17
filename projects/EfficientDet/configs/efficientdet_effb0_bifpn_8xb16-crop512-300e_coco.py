_base_ = [
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]
custom_imports = dict(
    imports=['projects.EfficientDet.efficientdet'], allow_failed_imports=False)

image_size = 512
batch_augments = [
    dict(type='BatchFixedSizePad', size=(image_size, image_size))
]
dataset_type = 'CocoDataset'
evalute_type = 'CocoMetric'
norm_cfg = dict(type='SyncBN', requires_grad=True, eps=1e-3, momentum=0.01)
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32-aa-advprop_in1k_20220119-26434485.pth'  # noqa
model = dict(
    type='EfficientDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=image_size,
        batch_augments=batch_augments),
    backbone=dict(
        type='EfficientNet',
        arch='b0',
        drop_path_rate=0.2,
        out_indices=(3, 4, 5),
        frozen_stages=0,
        conv_cfg=dict(type='Conv2dSamePadding'),
        norm_cfg=norm_cfg,
        norm_eval=False,
        init_cfg=dict(
            type='Pretrained', prefix='backbone', checkpoint=checkpoint)),
    neck=dict(
        type='BiFPN',
        num_stages=3,
        in_channels=[40, 112, 320],
        out_channels=64,
        start_level=0,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='EfficientDetSepBNHead',
        num_classes=80,
        num_ins=5,
        in_channels=64,
        feat_channels=64,
        stacked_convs=3,
        norm_cfg=norm_cfg,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0],
            strides=[8, 16, 32, 64, 128],
            center_offset=0.5),
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
        loss_bbox=dict(type='HuberLoss', beta=0.1, loss_weight=50)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        sampler=dict(
            type='PseudoSampler'),  # Focal loss should use PseudoSampler
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(
            type='soft_nms',
            iou_threshold=0.3,
            sigma=0.5,
            min_score=1e-3,
            method='gaussian'),
        max_per_img=100))

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(image_size, image_size),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(image_size, image_size)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(image_size, image_size), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(type=dataset_type, pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(type=dataset_type, pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type=evalute_type)
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(lr=0.16, weight_decay=4e-5),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
    clip_grad=dict(max_norm=10, norm_type=2))

# learning policy
max_epochs = 300
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=917),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0,
        begin=1,
        T_max=299,
        end=300,
        by_epoch=True,
        convert_to_iter_based=True)
]
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=15))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49)
]
# cudnn_benchmark=True can accelerate fix-size training
env_cfg = dict(cudnn_benchmark=True)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (16 samples per GPU)
auto_scale_lr = dict(base_batch_size=128)

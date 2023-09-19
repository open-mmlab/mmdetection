_base_ = [
    '../_base_/datasets/v3det.py', '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='FCOS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
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
        num_classes=13204,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        cls_predictor_cfg=dict(type='NormedLinear', tempearture=50, bias=True),
        loss_cls=dict(
            type='FocalCustomLoss',
            use_sigmoid=True,
            num_classes=13204,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            perm_repeat_gt_cfg=dict(iou_thr=0.7, perm_range=0.01)),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.0001,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=300))
# dataset settings

backend_args = None

train_dataloader = dict(batch_size=2, num_workers=8)

# training schedule for 2x
max_iter = 68760 * 2 * 2
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iter,
    val_interval=max_iter)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 2048,
        by_epoch=False,
        begin=0,
        end=5000 * 2),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[45840 * 2 * 2, 63030 * 2 * 2],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True, type='AdamW', lr=1e-4 * 0.25, weight_decay=0.1),
    clip_grad=dict(max_norm=35, norm_type=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5730 * 2))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

find_unused_parameters = True

_base_ = [
    '../_base_/models/cascade-rcnn_r50_fpn.py', '../_base_/datasets/v3det.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    rpn_head=dict(
        loss_bbox=dict(_delete_=True, type='L1Loss', loss_weight=1.0)),
    roi_head=dict(bbox_head=[
        dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=13204,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            cls_predictor_cfg=dict(
                type='NormedLinear', tempearture=50, bias=True),
            loss_cls=dict(
                type='CrossEntropyCustomLoss',
                num_classes=13204,
                use_sigmoid=True,
                loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=13204,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.05, 0.05, 0.1, 0.1]),
            reg_class_agnostic=True,
            cls_predictor_cfg=dict(
                type='NormedLinear', tempearture=50, bias=True),
            loss_cls=dict(
                type='CrossEntropyCustomLoss',
                num_classes=13204,
                use_sigmoid=True,
                loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=13204,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.033, 0.033, 0.067, 0.067]),
            reg_class_agnostic=True,
            cls_predictor_cfg=dict(
                type='NormedLinear', tempearture=50, bias=True),
            loss_cls=dict(
                type='CrossEntropyCustomLoss',
                num_classes=13204,
                use_sigmoid=True,
                loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))
    ]),
    # model training and testing settings
    train_cfg=dict(
        rpn_proposal=dict(nms_pre=4000, max_per_img=2000),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                    perm_repeat_gt_cfg=dict(iou_thr=0.7, perm_range=0.01)),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                    perm_repeat_gt_cfg=dict(iou_thr=0.7, perm_range=0.01)),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                    perm_repeat_gt_cfg=dict(iou_thr=0.7, perm_range=0.01)),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=300)))
# dataset settings
train_dataloader = dict(batch_size=4, num_workers=8)

# training schedule for 1x
max_iter = 68760 * 2
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
        end=5000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[45840 * 2, 63030 * 2],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(_delete_=True, type='AdamW', lr=1e-4 * 1, weight_decay=0.1),
    clip_grad=dict(max_norm=35, norm_type=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5730 * 2))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

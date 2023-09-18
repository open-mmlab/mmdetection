_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py', '../_base_/datasets/v3det.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=13204,
            reg_class_agnostic=True,
            cls_predictor_cfg=dict(
                type='NormedLinear', tempearture=50, bias=True),
            loss_cls=dict(
                type='CrossEntropyCustomLoss',
                num_classes=13204,
                use_sigmoid=True,
                loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn_proposal=dict(nms_pre=4000, max_per_img=2000),
        rcnn=dict(
            assigner=dict(
                perm_repeat_gt_cfg=dict(iou_thr=0.7, perm_range=0.01)))),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=300)))
# dataset settings
train_dataloader = dict(batch_size=4, num_workers=8)

# training schedule for 2x
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

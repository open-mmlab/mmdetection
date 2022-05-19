_base_ = ['../../../configs/_base_/schedules/schedule_1x.py', '../../../configs/_base_/default_runtime.py',
          '../../_base_/datasets/TinyPerson/TinyPerson_newData_detection_640x512.py', ]

# model settings
model = dict(
    type='YOLOX',
    backbone=dict(type='CSPDarknet', deepen_factor=1.0, widen_factor=1.0),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=10,  # 80
        in_channels=256,
        feat_channels=256),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(
        score_thr=0.01,
        nms=dict(type='soft_nms', iou_threshold=0.3, min_score=0.05),
    ))

img_scale = (640, 512)

# optimizer
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  # 5 epoch
    num_last_epochs=15,
    min_lr_ratio=0.05)
runner = dict(type='EpochBasedRunner', max_epochs=12)

resume_from = None
interval = 10

custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=2, priority=48),
    dict(
        type='SyncRandomSizeHook',
        ratio_range=(14, 26),
        img_scale=img_scale,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=2,
        interval=interval,
        priority=48),
    dict(type='ExpMomentumEMAHook', resume_from=resume_from, priority=49)
]
checkpoint_config = dict(interval=interval)
evaluation = dict(interval=interval, metric='bbox', save_best='bbox_mAP')
log_config = dict(interval=50)

_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/objects365v2_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(norm_cfg=dict(type='SyncBN', requires_grad=True)),
    roi_head=dict(bbox_head=dict(num_classes=365)))

# training schedule for 1350K
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=1350000,  # 36 epochs
    val_interval=150000)

# Using 8 GPUS while training
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 1000,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=1350000,
        by_epoch=False,
        milestones=[900000, 1200000],
        gamma=0.1)
]

train_dataloader = dict(sampler=dict(type='InfiniteSampler'))
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=150000))

log_processor = dict(by_epoch=False)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)

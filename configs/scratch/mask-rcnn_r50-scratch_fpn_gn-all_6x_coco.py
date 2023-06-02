_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    backbone=dict(
        frozen_stages=-1,
        zero_init_residual=False,
        norm_cfg=norm_cfg,
        init_cfg=None),
    neck=dict(norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg),
        mask_head=dict(norm_cfg=norm_cfg)))

optim_wrapper = dict(paramwise_cfg=dict(norm_decay_mult=0.))

max_epochs = 73

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[65, 71],
        gamma=0.1)
]

train_cfg = dict(max_epochs=max_epochs)

# only keep latest 3 checkpoints
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))

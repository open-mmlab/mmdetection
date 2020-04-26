_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    pretrained=None,
    backbone=dict(
        frozen_stages=-1, zero_init_residual=False, norm_cfg=norm_cfg),
    neck=dict(norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg),
        mask_head=dict(norm_cfg=norm_cfg)))
# optimizer
optimizer = dict(paramwise_cfg=dict(norm_decay_mult=0))
optimizer_config = dict(_delete_=True, grad_clip=None)
# learning policy
lr_config = dict(warmup_ratio=0.1, step=[65, 71])
total_epochs = 73

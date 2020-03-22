_base_ = [
    '../component/faster_rcnn_r50_fpn.py', '../component/coco_detection.py',
    '../component/default_runtime.py'
]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    pretrained=None,
    backbone=dict(
        frozen_stages=-1, zero_init_residual=False, norm_cfg=norm_cfg),
    neck=dict(norm_cfg=norm_cfg),
    bbox_head=dict(
        type='Shared4Conv1FCBBoxHead',
        conv_out_channels=256,
        norm_cfg=norm_cfg))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.02,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_options=dict(norm_decay_mult=0))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[65, 71])
total_epochs = 73
work_dir = './work_dirs/scratch_faster_rcnn_r50_fpn_gn_6x'

_base_ = './grid_rcnn_gn_head_r50_fpn_2x.py'
model = dict(
    pretrained='open-mmlab://resnext101_32x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=3665,
    warmup_ratio=1.0 / 80,
    step=[17, 23])
total_epochs = 25
work_dir = './work_dirs/grid_rcnn_gn_head_x101_32x4d_fpn_2x'

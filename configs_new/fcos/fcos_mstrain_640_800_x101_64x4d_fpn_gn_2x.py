_base_ = './fcos_r50_fpn_gn_1x_4gpu.py'
model = dict(
    pretrained='open-mmlab://resnext101_64x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'))
data = dict(imgs_per_gpu=2, workers_per_gpu=2)
# learning policy
lr_config = dict(step=[16, 22])
total_epochs = 24
work_dir = './work_dirs/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x'

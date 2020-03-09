_base_ = './reppoints_moment_r50_fpn_2x_mt.py'
model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        dcn=dict(
            modulated=False, deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
work_dir = './work_dirs/reppoints_moment_r101_dcn_fpn_2x_mt'

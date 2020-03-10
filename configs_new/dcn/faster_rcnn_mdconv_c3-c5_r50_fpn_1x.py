_base_ = '../faster_rcnn_r50_fpn_1x.py'
model = dict(
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
work_dir = './work_dirs/faster_rcnn_mdconv_c3-c5_r50_fpn_1x'

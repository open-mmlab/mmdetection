_base_ = './mask_rcnn_r50_fpn_gn_2x.py'
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    pretrained='open-mmlab://detectron/resnet101_gn',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        norm_cfg=norm_cfg))
work_dir = './work_dirs/mask_rcnn_r101_fpn_gn_2x'

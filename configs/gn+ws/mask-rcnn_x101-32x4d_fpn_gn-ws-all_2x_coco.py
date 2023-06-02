_base_ = './mask-rcnn_r50_fpn_gn-ws-all_2x_coco.py'
# model settings
conv_cfg = dict(type='ConvWS')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://jhu/resnext101_32x4d_gn_ws')))

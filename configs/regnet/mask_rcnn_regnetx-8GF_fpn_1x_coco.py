_base_ = './mask_rcnn_regnetx_3GF_fpn_1x_coco.py'
model = dict(
    pretrained='open-mmlab://regnetx_8.0gf',
    backbone=dict(
        type='RegNet',
        depth=23,
        arch_parameter=dict(
            w0=80, wa=49.56, wm=2.88, group_w=120, bot_mul=1.0),
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[80, 240, 720, 1920],
        out_channels=256,
        num_outs=5))

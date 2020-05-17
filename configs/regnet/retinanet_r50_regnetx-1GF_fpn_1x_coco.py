_base_ = './retinanet_r50_regnetx-3GF_fpn_1x_coco.py'
model = dict(
    pretrained='open-mmlab://regnetx_1.6gf',
    backbone=dict(
        type='RegNet',
        depth=18,
        arch_parameter=dict(w0=80, wa=34.01, wm=2.25, group_w=24, bot_mul=1.0),
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[72, 168, 408, 912],
        out_channels=256,
        num_outs=5))

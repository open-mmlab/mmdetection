_base_ = './retinanet_regnetx-3.2GF_fpn_1x_coco.py'
model = dict(
    pretrained='open-mmlab://regnetx_800mf',
    backbone=dict(
        type='RegNet',
        arch='regnetx_800mf',
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 288, 672],
        out_channels=256,
        num_outs=5))

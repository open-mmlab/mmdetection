_base_ = './mask_rcnn_regnetx-3GF_fpn_1x_coco.py'
model = dict(
    pretrained='open-mmlab://regnetx_12gf',
    backbone=dict(
        type='RegNet',
        arch='regnetx_12gf',
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[224, 448, 896, 2240],
        out_channels=256,
        num_outs=5))

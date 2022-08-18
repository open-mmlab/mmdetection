_base_ = 'cascade_mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco.py'
model = dict(
    backbone=dict(
        type='RegNet',
        arch='regnetx_1.6gf',
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://regnetx_1.6gf')),
    neck=dict(
        type='FPN',
        in_channels=[72, 168, 408, 912],
        out_channels=256,
        num_outs=5))

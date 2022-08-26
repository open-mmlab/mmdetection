_base_ = 'faster-rcnn_regnetx-3.2GF_fpn_ms-3x_coco.py'
model = dict(
    backbone=dict(
        type='RegNet',
        arch='regnetx_4.0gf',
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://regnetx_4.0gf')),
    neck=dict(
        type='FPN',
        in_channels=[80, 240, 560, 1360],
        out_channels=256,
        num_outs=5))

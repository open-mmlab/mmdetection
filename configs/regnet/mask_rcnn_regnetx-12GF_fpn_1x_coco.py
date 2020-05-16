_base_ = './mask_rcnn_regnetx_3GF_fpn_1x_coco.py'
model = dict(
    pretrained='./regnet_pretrain/RegNetX-12GF.pth',
    backbone=dict(
        type='RegNet',
        depth=19,
        arch_parameter=dict(
            w0=168, wa=73.36, wm=2.37, group_w=112, bot_mul=1.0),
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

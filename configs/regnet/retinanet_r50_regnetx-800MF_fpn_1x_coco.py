_base_ = './retinanet_r50_regnetx-3GF_fpn_1x_coco.py'
model = dict(
    pretrained='./regnet_pretrain/RegNetX-800MF.pth',
    backbone=dict(
        type='RegNet',
        depth=16,
        arch_parameter=dict(w0=56, wa=35.73, wm=2.28, group_w=16, bot_mul=1.0),
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

_base_ = './mask_rcnn_regnetx_3GF_fpn_1x_coco.py'
model = dict(
    pretrained='./regnet_pretrain/RegNetX-4.0GF.pth',
    backbone=dict(
        type='RegNet',
        depth=23,
        arch_parameter=dict(w0=96, wa=38.65, wm=2.43, group_w=40, bot_mul=1.0),
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[80, 240, 560, 1360],
        out_channels=256,
        num_outs=5))

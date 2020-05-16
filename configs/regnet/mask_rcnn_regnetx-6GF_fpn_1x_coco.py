_base_ = './mask_rcnn_regnetx_3GF_fpn_1x_coco.py'
model = dict(
    pretrained='./regnet_pretrain/RegNetX-6.4GF.pth',
    backbone=dict(
        type='RegNet',
        depth=17,
        arch_parameter=dict(
            w0=184, wa=60.83, wm=2.07, group_w=56, bot_mul=1.0),
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[168, 392, 784, 1624],
        out_channels=256,
        num_outs=5))

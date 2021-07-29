_base_ = ['retinanet_pvt_v2_b0_fpn_1x_coco.py']
model = dict(
    pretrained='https://github.com/whai362/PVT/'
    'releases/download/v2/pvt_v2_b1.pth',
    backbone=dict(embed_dims=64),
    neck=dict(in_channels=[64, 128, 320, 512]))

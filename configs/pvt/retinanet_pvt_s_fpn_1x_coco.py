_base_ = [
    'retinanet_pvt_t_fpn_1x_coco.py',
]
model = dict(
    pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_small.pth',
    backbone=dict(num_layers=[3, 4, 6, 3]))

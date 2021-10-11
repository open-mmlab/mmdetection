_base_ = 'retinanet_pvt-t_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        num_layers=[3, 4, 18, 3],
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
                      'releases/download/v2/pvt_medium.pth')))

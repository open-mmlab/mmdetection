if '_base_':
    from .retinanet_pvt_t_fpn_1x_coco import *

model.merge(
    dict(
        backbone=dict(
            num_layers=[3, 4, 6, 3],
            init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
                          'releases/download/v2/pvt_small.pth'))))

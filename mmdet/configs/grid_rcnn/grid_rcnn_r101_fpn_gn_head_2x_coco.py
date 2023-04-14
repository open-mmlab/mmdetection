if '_base_':
    from .grid_rcnn_r50_fpn_gn_head_2x_coco import *

model.merge(
    dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101'))))

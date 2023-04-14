if '_base_':
    from ..common.ms_poly_3x_coco_instance import *
    from .._base_.models.mask_rcnn_r50_fpn import *

model.merge(
    dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101'))))

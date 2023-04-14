if '_base_':
    from .htc_x101_32x4d_fpn_16xb1_20e_coco import *
from mmdet.models.backbones.resnext import ResNeXt

model.merge(
    dict(
        backbone=dict(
            type=ResNeXt,
            groups=64,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://resnext101_64x4d'))))

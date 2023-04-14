if '_base_':
    from .cascade_rcnn_r50_fpn_1x_coco import *
from mmdet.models.detectors.cascade_rcnn import CascadeRCNN
from mmdet.models.backbones.resnext import ResNeXt

model.merge(
    dict(
        type=CascadeRCNN,
        backbone=dict(
            type=ResNeXt,
            depth=101,
            groups=64,
            base_width=4,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            style='pytorch',
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://resnext101_64x4d'))))

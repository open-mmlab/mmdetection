if '_base_':
    from .gfl_r50_fpn_ms_2x_coco import *
from mmdet.models.backbones.resnet import ResNet

model.merge(
    dict(
        backbone=dict(
            type=ResNet,
            depth=101,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101'))))

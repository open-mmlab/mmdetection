if '_base_':
    from .mask_rcnn_regnetx_3_2GF_fpn_1x_coco import *
from mmdet.models.backbones.regnet import RegNet
from mmdet.models.necks.fpn import FPN

model.merge(
    dict(
        backbone=dict(
            type=RegNet,
            arch='regnetx_12gf',
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://regnetx_12gf')),
        neck=dict(
            type=FPN,
            in_channels=[224, 448, 896, 2240],
            out_channels=256,
            num_outs=5)))

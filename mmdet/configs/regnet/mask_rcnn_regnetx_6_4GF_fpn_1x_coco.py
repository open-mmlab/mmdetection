if '_base_':
    from .mask_rcnn_regnetx_3_2GF_fpn_1x_coco import *
from mmdet.models.backbones.regnet import RegNet
from mmdet.models.necks.fpn import FPN

model.merge(
    dict(
        backbone=dict(
            type=RegNet,
            arch='regnetx_6.4gf',
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://regnetx_6.4gf')),
        neck=dict(
            type=FPN,
            in_channels=[168, 392, 784, 1624],
            out_channels=256,
            num_outs=5)))

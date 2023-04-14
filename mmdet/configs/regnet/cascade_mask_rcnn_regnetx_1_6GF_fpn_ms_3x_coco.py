if '_base_':
    from .cascade_mask_rcnn_regnetx_3_2GF_fpn_ms_3x_coco import *
from mmdet.models.backbones.regnet import RegNet
from mmdet.models.necks.fpn import FPN

model.merge(
    dict(
        backbone=dict(
            type=RegNet,
            arch='regnetx_1.6gf',
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://regnetx_1.6gf')),
        neck=dict(
            type=FPN,
            in_channels=[72, 168, 408, 912],
            out_channels=256,
            num_outs=5)))

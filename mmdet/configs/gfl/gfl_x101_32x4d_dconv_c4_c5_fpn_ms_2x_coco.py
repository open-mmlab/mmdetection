if '_base_':
    from .gfl_r50_fpn_ms_2x_coco import *
from mmdet.models.detectors.gfl import GFL
from mmdet.models.backbones.resnext import ResNeXt

model.merge(
    dict(
        type=GFL,
        backbone=dict(
            type=ResNeXt,
            depth=101,
            groups=32,
            base_width=4,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://resnext101_32x4d'))))

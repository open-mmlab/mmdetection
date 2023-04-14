if '_base_':
    from .vfnet_r50_mdconv_c3_c5_fpn_ms_2x_coco import *
from mmdet.models.backbones.res2net import Res2Net

model.merge(
    dict(
        backbone=dict(
            type=Res2Net,
            depth=101,
            scales=4,
            base_width=26,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, True, True, True),
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://res2net101_v1d_26w_4s'))))

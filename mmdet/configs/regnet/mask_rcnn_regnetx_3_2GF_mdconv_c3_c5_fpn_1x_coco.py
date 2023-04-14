if '_base_':
    from .mask_rcnn_regnetx_3_2GF_fpn_1x_coco import *

model.merge(
    dict(
        backbone=dict(
            dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, True, True, True),
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://regnetx_3.2gf'))))

if '_base_':
    from .reppoints_moment_r50_fpn_gn_head_gn_2x_coco import *

model.merge(
    dict(
        backbone=dict(
            depth=101,
            dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, True, True, True),
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101'))))

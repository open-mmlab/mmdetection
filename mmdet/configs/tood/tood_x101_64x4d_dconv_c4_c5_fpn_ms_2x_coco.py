if '_base_':
    from .tood_x101_64x4d_fpn_ms_2x_coco import *

model.merge(
    dict(
        backbone=dict(
            dcn=dict(
                type='DCNv2', deformable_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True),
        ),
        bbox_head=dict(num_dcn=2)))

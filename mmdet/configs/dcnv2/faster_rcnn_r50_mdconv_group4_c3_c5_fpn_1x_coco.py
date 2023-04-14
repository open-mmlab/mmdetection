if '_base_':
    from ..faster_rcnn.faster_rcnn_r50_fpn_1x_coco import *

model.merge(
    dict(
        backbone=dict(
            dcn=dict(type='DCNv2', deform_groups=4, fallback_on_stride=False),
            stage_with_dcn=(False, True, True, True))))

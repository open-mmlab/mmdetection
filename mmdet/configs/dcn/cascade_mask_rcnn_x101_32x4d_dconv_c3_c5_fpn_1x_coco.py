if '_base_':
    from ..cascade_rcnn.cascade_mask_rcnn_x101_32x4d_fpn_1x_coco import *

model.merge(
    dict(
        backbone=dict(
            dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, True, True, True))))

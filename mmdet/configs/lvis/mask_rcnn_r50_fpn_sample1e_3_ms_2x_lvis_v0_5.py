if '_base_':
    from .._base_.models.mask_rcnn_r50_fpn import *
    from .._base_.datasets.lvis_v0_5_instance import *
    from .._base_.schedules.schedule_2x import *
    from .._base_.default_runtime import *

model.merge(
    dict(
        roi_head=dict(
            bbox_head=dict(num_classes=1230),
            mask_head=dict(num_classes=1230)),
        test_cfg=dict(
            rcnn=dict(
                score_thr=0.0001,
                # LVIS allows up to 300
                max_per_img=300))))

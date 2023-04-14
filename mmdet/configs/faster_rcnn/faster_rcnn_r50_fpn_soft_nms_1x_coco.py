if '_base_':
    from .._base_.models.faster_rcnn_r50_fpn import *
    from .._base_.datasets.coco_detection import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *

model.merge(
    dict(
        test_cfg=dict(
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='soft_nms', iou_threshold=0.5),
                max_per_img=100))))

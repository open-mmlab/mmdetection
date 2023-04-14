if '_base_':
    from .reppoints_moment_r50_fpn_gn_head_gn_1x_coco import *
from mmdet.models.task_modules.assigners.max_iou_assigner import MaxIoUAssigner

model.merge(
    dict(
        bbox_head=dict(transform_method='minmax', use_grid_points=True),
        # training and testing settings
        train_cfg=dict(
            init=dict(
                assigner=dict(
                    _delete_=True,
                    type=MaxIoUAssigner,
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1)))))

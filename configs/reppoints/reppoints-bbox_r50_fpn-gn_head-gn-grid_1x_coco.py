_base_ = './reppoints-moment_r50_fpn-gn_head-gn_1x_coco.py'
model = dict(
    bbox_head=dict(transform_method='minmax', use_grid_points=True),
    # training and testing settings
    train_cfg=dict(
        init=dict(
            assigner=dict(
                _delete_=True,
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1))))

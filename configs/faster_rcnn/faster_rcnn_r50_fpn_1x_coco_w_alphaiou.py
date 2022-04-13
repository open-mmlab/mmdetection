_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model
model = dict(
    rpn_head=dict(
        loss_bbox=dict(type='AlphaIoULoss', loss_weight=1.0)
    ),
    roi_head=dict(
        bbox_head=dict(
            loss_bbox=dict(type='AlphaIoULoss', loss_weight=1.0)
        )
    )
)
# data
data = dict(samples_per_gpu=4)
# runtime
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)
# schedule
optimizer = dict(lr=0.005)
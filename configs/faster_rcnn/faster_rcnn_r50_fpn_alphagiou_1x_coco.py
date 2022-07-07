_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model
model = dict(
    roi_head=dict(
        bbox_head=dict(
            reg_decoded_bbox=True,
            loss_bbox=dict(type='AlphaIoULoss', loss_weight=10.0, eps=1e-9))))
# data
data = dict(samples_per_gpu=4)
# runtime
log_config = dict(
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# Set evaluation interval
evaluation = dict(interval=2)
# Set checkpoint interval
checkpoint_config = dict(interval=4)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='NeptuneHook',
             project="mmdetection")
        ])

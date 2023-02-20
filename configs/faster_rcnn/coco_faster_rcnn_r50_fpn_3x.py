_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4
)